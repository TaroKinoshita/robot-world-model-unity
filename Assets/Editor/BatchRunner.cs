using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// P6: バッチ実行。seed を変えながら
///   Play → (EpisodeRecorder が 1 ペア作る) → Play を止める → Python 4 本(run_pipeline.py)
/// を無人でくり返す。
///
/// - 値(seed・sceneIndex・pairIndex・outputRoot)は Temp/p6_batch/job.json 経由で Play 中のインスタンスにだけ渡す
///   → シーンファイルは変わらない
/// - 生成に失敗した seed は飛ばして(scene / pair の番号は詰める)、ログに残す。途中まで書かれたフォルダは &lt;root&gt;/_failed/ へ移す(消さない)
/// - 状態は Temp/p6_batch/state.json に書く(Play の開始・終了のドメインリロードをまたいで続く)
/// - ログ: &lt;root&gt;/_batch/batch_log.txt、Python の出力: &lt;root&gt;/_batch/logs/pair_XXXX.txt
///
/// 使い方:
///   1. Episodes/_batch_config.json を書く(無ければ Tools > P6 Batch > Start で雛形ができる)
///   2. Tools > P6 Batch > Start
///   止める: Tools > P6 Batch > Stop after current(今のシーンが終わったら止める)/ Abort now(すぐ止める)
/// </summary>
[InitializeOnLoad]
public static class BatchRunner
{
    [Serializable]
    public class Config
    {
        public string outputRoot = "Episodes";   // プロジェクトからの相対パス、または絶対パス
        public int numScenes = 3;                // 成功させるペア(= Task A/B の 1 シーン)の数
        public int seedStart = 1;
        public int[] seeds = new int[0];         // 空でなければ、この順に使う(seedStart は無視)
        public int maxSeedAttempts = 0;          // 0 なら numScenes * 2 + 5
        public int sceneIndexStart = 0;          // 最初の Task A の番号(B は +1)
        public int pairIndexStart = 0;
        public float playTimeoutMin = 30f;       // 1 Play の上限(分)
        public float pythonTimeoutMin = 30f;     // run_pipeline.py の上限(分)
        public string pythonExe = @"C:\Users\kinos\AppData\Local\Programs\Python\Python311\python.exe";
        public bool requireCleanGit = true;      // 正式データは clean な commit から。テストだけ false にする
        public bool runPython = true;
    }

    [Serializable]
    public class JobRecord
    {
        public string jobId = "";
        public int seed, sceneIndex, pairIndex;
        public string status = "";               // ok / failed
        public string stage = "", reason = "";
        public int candidatesPlanned, candidatesExecuted;
        public string startLocal = "", endLocal = "";
        public float playMin, pythonMin;
        public string python = "";               // PASS / FAIL / skipped / timeout
        public string failedMovedTo = "";
    }

    [Serializable]
    public class State
    {
        public bool active;
        public string phase = "idle";            // next → enterPlay → playing → stopping → python → next … → done
        public Config config = new Config();
        public string rootAbs = "", batchDir = "", gitHead = "";
        public bool gitDirty;
        public int seedCursor, attempts, succeeded;
        public int curSeed, curScene, curPair;
        public string curJobId = "";
        public double phaseStart, jobStart, pyStart;
        public bool lastPlayOk;
        public int pythonPid;
        public bool stopRequested;
        public string startedLocal = "", finishedLocal = "", endReason = "";
        public List<JobRecord> records = new List<JobRecord>();
    }

    static State st;
    static double pausedSince = -1;

    static string ProjectDir => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
    static string StatePath => Path.Combine(BatchJob.Dir, "state.json");
    static string ConfigPath => Path.Combine(ProjectDir, "Episodes", "_batch_config.json");
    static double Now => (DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds;
    static string Local => DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

    static BatchRunner()
    {
        st = LoadState();
        EditorApplication.update += Tick;
    }

    // ================= メニュー / 外から呼ぶ =================

    [MenuItem("Tools/P6 Batch/Start")]
    static void MenuStart()
    {
        if (!File.Exists(ConfigPath))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ConfigPath));
            File.WriteAllText(ConfigPath, JsonUtility.ToJson(new Config(), true), new UTF8Encoding(false));
            Debug.LogWarning($"[BatchRunner] 設定ファイルが無いので雛形を作った。中身を確認してからもう一度 Start: {ConfigPath}");
            return;
        }
        Debug.Log("[BatchRunner] " + StartBatch(File.ReadAllText(ConfigPath, Encoding.UTF8)));
    }

    [MenuItem("Tools/P6 Batch/Stop after current")]
    public static void StopAfterCurrent()
    {
        if (st == null || !st.active) { Debug.Log("[BatchRunner] 動いていない"); return; }
        st.stopRequested = true; Save();
        Log("Stop after current を受け付けた(今のシーンが終わったら止める)");
    }

    [MenuItem("Tools/P6 Batch/Abort now")]
    public static void AbortNow()
    {
        if (st == null || !st.active) { Debug.Log("[BatchRunner] 動いていない"); return; }
        Finish("abort(手で止めた)");
        if (EditorApplication.isPlaying) EditorApplication.ExitPlaymode();
    }

    [MenuItem("Tools/P6 Batch/Status")]
    static void MenuStatus() { Debug.Log("[BatchRunner] " + Status()); }

    /// configJson: Config の JSON。戻り値は開始できたかの説明
    public static string StartBatch(string configJson)
    {
        if (st != null && st.active) return $"もう動いている(phase={st.phase})。止めるなら Abort now";
        if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode) return "Play 中は始められない";
        if (EditorApplication.isCompiling) return "コンパイル中。終わってから";

        Config cfg;
        try { cfg = JsonUtility.FromJson<Config>(configJson); }
        catch (Exception e) { return "設定を読めない: " + e.Message; }
        if (cfg == null || cfg.numScenes <= 0) return "numScenes が 0";
        if (cfg.maxSeedAttempts <= 0) cfg.maxSeedAttempts = cfg.numScenes * 2 + 5;

        string root = Path.IsPathRooted(cfg.outputRoot) ? cfg.outputRoot : Path.Combine(ProjectDir, cfg.outputRoot);
        root = Path.GetFullPath(root);

        var er = UnityEngine.Object.FindFirstObjectByType<EpisodeRecorder>();
        if (er == null || !er.enabled) return "シーンに有効な EpisodeRecorder が無い";
        if (!er.generateTaskPair || !er.autoScene.enabled) return "EpisodeRecorder の generateTaskPair と autoScene.enabled を on にして";
        if (er.debugOnlyCandidates != null && er.debugOnlyCandidates.Length > 0) return "debugOnlyCandidates が空じゃない(本番は空)";
        var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if (scene.isDirty && cfg.requireCleanGit) return "シーンに保存していない変更がある";
        if (cfg.runPython && !File.Exists(cfg.pythonExe)) return "python が見つからない: " + cfg.pythonExe;

        // git(clean な commit から作る)
        string head = Git("rev-parse --short HEAD").Trim();
        string porcelain = Git("status --porcelain");
        bool dirty = porcelain.Trim().Length > 0;
        if (dirty && cfg.requireCleanGit) return "作業フォルダが clean じゃない(commit してから):\n" + porcelain;

        // 最初のペアの番号が空いているか
        string clash = FindClash(root, cfg.sceneIndexStart, cfg.pairIndexStart);
        if (clash != null) return "出力先にもうある: " + clash + "(sceneIndexStart / pairIndexStart を空いている番号にして)";

        Directory.CreateDirectory(BatchJob.Dir);
        try { if (File.Exists(BatchJob.JobPath)) File.Delete(BatchJob.JobPath); if (File.Exists(BatchJob.ResultPath)) File.Delete(BatchJob.ResultPath); } catch { }

        st = new State
        {
            active = true, phase = "next", config = cfg, rootAbs = root,
            batchDir = Path.Combine(root, "_batch"), gitHead = head, gitDirty = dirty,
            startedLocal = Local, phaseStart = Now
        };
        Directory.CreateDirectory(st.batchDir);
        Save();
        Log($"===== batch 開始: {cfg.numScenes} シーン、seed {(cfg.seeds.Length > 0 ? string.Join(",", cfg.seeds) : cfg.seedStart + "〜")}、" +
            $"scene {cfg.sceneIndexStart}〜 / pair {cfg.pairIndexStart}〜、出力 {root}、git {head}{(dirty ? "(dirty)" : "")}");
        return "開始した: " + root;
    }

    public static string Status()
    {
        if (st == null) return "state なし";
        var ok = st.records.Count(r => r.status == "ok");
        var pyPass = st.records.Count(r => r.python == "PASS");
        return $"active={st.active} phase={st.phase} succeeded={st.succeeded}/{st.config.numScenes} attempts={st.attempts} " +
               $"cur seed={st.curSeed} scene={st.curScene} pair={st.curPair} records ok={ok} pythonPASS={pyPass} end='{st.endReason}' root={st.rootAbs}";
    }

    // ================= 本体(EditorApplication.update) =================

    static void Tick()
    {
        if (st == null || !st.active) return;
        try { Step(); }
        catch (Exception e)
        {
            Log("例外で止めた: " + e);
            Finish("exception: " + e.Message);
        }
    }

    static void Step()
    {
        switch (st.phase)
        {
            case "next": StepNext(); break;
            case "enterPlay": StepEnterPlay(); break;
            case "playing": StepPlaying(); break;
            case "stopping": StepStopping(); break;
            case "python": StepPython(); break;
        }
    }

    static void SetPhase(string p) { st.phase = p; st.phaseStart = Now; Save(); }

    static void StepNext()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling || EditorApplication.isUpdating) return;
        var cfg = st.config;
        if (st.succeeded >= cfg.numScenes) { Finish("完了"); return; }
        if (st.stopRequested) { Finish("Stop after current で止めた"); return; }
        if (st.attempts >= cfg.maxSeedAttempts) { Finish($"seed を {st.attempts} 回試したので打ち切り"); return; }
        if (cfg.seeds.Length > 0 && st.seedCursor >= cfg.seeds.Length) { Finish("seeds を使い切った"); return; }

        st.curSeed = cfg.seeds.Length > 0 ? cfg.seeds[st.seedCursor] : cfg.seedStart + st.seedCursor;
        st.seedCursor++;
        st.attempts++;
        st.curScene = cfg.sceneIndexStart + 2 * st.succeeded;
        st.curPair = cfg.pairIndexStart + st.succeeded;
        string clash = FindClash(st.rootAbs, st.curScene, st.curPair);
        if (clash != null) { Finish("出力先にもうある: " + clash); return; }

        st.curJobId = $"seed{st.curSeed}_{DateTime.Now:yyyyMMdd_HHmmss}";
        try { if (File.Exists(BatchJob.ResultPath)) File.Delete(BatchJob.ResultPath); } catch { }
        var job = new BatchJob.Job { jobId = st.curJobId, seed = st.curSeed, sceneIndex = st.curScene, pairIndex = st.curPair, outputRoot = st.rootAbs };
        Directory.CreateDirectory(BatchJob.Dir);
        File.WriteAllText(BatchJob.JobPath, JsonUtility.ToJson(job, true), new UTF8Encoding(false));
        st.jobStart = Now;
        st.records.Add(new JobRecord { jobId = st.curJobId, seed = st.curSeed, sceneIndex = st.curScene, pairIndex = st.curPair, startLocal = Local });
        Log($"--- [{st.attempts}] seed {st.curSeed} → scene_{st.curScene:D4}/{st.curScene + 1:D4}, pair_{st.curPair:D4}: Play");
        SetPhase("enterPlay");
        EditorApplication.EnterPlaymode();
    }

    static void StepEnterPlay()
    {
        if (EditorApplication.isPlaying) { pausedSince = -1; SetPhase("playing"); return; }
        if (Now - st.phaseStart > 120)
        {
            // コンパイルエラーなどで Play に入れない
            Log("2 分待っても Play に入らない(コンパイルエラー?)。batch を止める");
            Cur().status = "failed"; Cur().stage = "enterPlay"; Cur().reason = "Play に入れない";
            Finish("Play に入れない");
        }
    }

    static void StepPlaying()
    {
        if (!EditorApplication.isPlaying && !EditorApplication.isPlayingOrWillChangePlaymode)
        {
            // 結果を書く前に Play が終わった = 誰かが止めた
            var r0 = ReadResult();
            if (r0 == null)
            {
                Cur().status = "failed"; Cur().stage = "play"; Cur().reason = "結果の前に Play が止まった(手で止めた?)";
                Log("結果の前に Play が止まった。手で止めたとみなして batch も止める");
                Finish("Play が外から止められた");
                return;
            }
        }

        var r = ReadResult();
        if (r != null)
        {
            var rec = Cur();
            rec.status = r.ok ? "ok" : "failed"; rec.stage = r.stage; rec.reason = r.reason;
            rec.candidatesPlanned = r.candidatesPlanned; rec.candidatesExecuted = r.candidatesExecuted;
            rec.playMin = (float)((Now - st.jobStart) / 60.0);
            st.lastPlayOk = r.ok;
            Log(r.ok ? $"Play 完了({rec.playMin:F1} 分、候補 {r.candidatesExecuted} 本実行)。Play を止める"
                     : $"失敗: stage={r.stage} reason={r.reason}。Play を止めて次の seed へ");
            SetPhase("stopping");
            EditorApplication.ExitPlaymode();
            return;
        }

        // Error Pause などで止まったまま
        if (EditorApplication.isPaused)
        {
            if (pausedSince < 0) pausedSince = Now;
            else if (Now - pausedSince > 10)
            {
                FailCurrent("paused", "Play が一時停止した(LogError の Error Pause / 例外)。コンソール参照");
                return;
            }
        }
        else pausedSince = -1;

        if (Now - st.jobStart > st.config.playTimeoutMin * 60)
            FailCurrent("timeout", $"{st.config.playTimeoutMin} 分たっても終わらない");
    }

    static void FailCurrent(string stage, string reason)
    {
        var rec = Cur();
        rec.status = "failed"; rec.stage = stage; rec.reason = reason;
        rec.playMin = (float)((Now - st.jobStart) / 60.0);
        st.lastPlayOk = false;
        Log($"失敗: {stage}: {reason}。Play を止めて次の seed へ");
        SetPhase("stopping");
        EditorApplication.isPaused = false;
        EditorApplication.ExitPlaymode();
    }

    static void StepStopping()
    {
        if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
        {
            if (Now - st.phaseStart > 60) { EditorApplication.isPaused = false; EditorApplication.ExitPlaymode(); st.phaseStart = Now; }
            return;
        }
        // SOLO の annotation_definitions.json は Play を止めたときに書かれる。少し待つ
        if (Now - st.phaseStart < 3) return;

        var rec = Cur();
        rec.endLocal = Local;
        if (!st.lastPlayOk)
        {
            rec.python = "skipped";
            rec.failedMovedTo = MoveFailed(rec);
            Log($"seed {rec.seed} は飛ばした(stage={rec.stage})" + (rec.failedMovedTo != "" ? $"。途中のフォルダは {rec.failedMovedTo} へ移した" : ""));
            SetPhase("next");
            return;
        }

        if (!st.config.runPython)
        {
            rec.python = "skipped";
            st.succeeded++;
            SetPhase("next");
            return;
        }

        // Python 4 本を別プロセスで
        string logDir = Path.Combine(st.batchDir, "logs");
        Directory.CreateDirectory(logDir);
        string pair = $"pair_{st.curPair:D4}";
        string resultJson = Path.Combine(logDir, pair + ".json");
        try { if (File.Exists(resultJson)) File.Delete(resultJson); } catch { }
        string script = Path.Combine(Application.dataPath, "Python", "run_pipeline.py");
        string args = $"\"{script}\" --episodes \"{st.rootAbs}\" --scenes scene_{st.curScene:D4} scene_{st.curScene + 1:D4} --pair {pair} " +
                      $"--log \"{Path.Combine(logDir, pair + ".txt")}\" --result \"{resultJson}\"";
        var psi = new ProcessStartInfo(st.config.pythonExe, args)
        {
            WorkingDirectory = ProjectDir, UseShellExecute = false, CreateNoWindow = true
        };
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["OPENCV_IO_ENABLE_OPENEXR"] = "1";
        var p = Process.Start(psi);
        st.pythonPid = p != null ? p.Id : 0;
        st.pyStart = Now;
        Log($"Python 開始(pid {st.pythonPid}): {pair}");
        SetPhase("python");
    }

    static void StepPython()
    {
        string pair = $"pair_{st.curPair:D4}";
        string resultJson = Path.Combine(st.batchDir, "logs", pair + ".json");
        var rec = Cur();
        if (File.Exists(resultJson))
        {
            string txt;
            try { txt = File.ReadAllText(resultJson, Encoding.UTF8); } catch { return; }
            bool pass = txt.Contains("\"pass\": true");
            rec.python = pass ? "PASS" : "FAIL";
            rec.pythonMin = (float)((Now - st.pyStart) / 60.0);
            st.succeeded++;
            Log($"Python {rec.python}({rec.pythonMin:F1} 分): {pair}  → 成功 {st.succeeded}/{st.config.numScenes}" +
                (pass ? "" : $"。詳しくは {Path.Combine(st.batchDir, "logs", pair + ".txt")}"));
            SetPhase("next");
            return;
        }
        if (!ProcessAlive(st.pythonPid) && Now - st.pyStart > 5)
        {
            // 結果を書かずに終わった(起動失敗・クラッシュ)
            if (Now - st.pyStart < 15) return;   // 書き終わり待ちの猶予
            rec.python = "FAIL"; rec.pythonMin = (float)((Now - st.pyStart) / 60.0);
            st.succeeded++;
            Log($"Python が結果を書かずに終わった: {pair}(データは残す。手で run_pipeline.py を回して確認)");
            SetPhase("next");
            return;
        }
        if (Now - st.pyStart > st.config.pythonTimeoutMin * 60)
        {
            try { Process.GetProcessById(st.pythonPid).Kill(); } catch { }
            rec.python = "timeout"; rec.pythonMin = (float)((Now - st.pyStart) / 60.0);
            st.succeeded++;
            Log($"Python が {st.config.pythonTimeoutMin} 分で終わらないので止めた: {pair}(データは残す)");
            SetPhase("next");
        }
    }

    static void Finish(string why)
    {
        st.active = false;
        st.phase = "done";
        st.endReason = why;
        st.finishedLocal = Local;
        Save();
        int ok = st.records.Count(r => r.status == "ok");
        int failed = st.records.Count(r => r.status == "failed");
        int pass = st.records.Count(r => r.python == "PASS");
        Log($"===== batch 終了: {why}。成功 {st.succeeded}/{st.config.numScenes}、Play ok {ok}・失敗 {failed}、Python PASS {pass}");
        foreach (var r in st.records)
            Log($"  seed {r.seed,4}  scene_{r.sceneIndex:D4}  pair_{r.pairIndex:D4}  {r.status,-6} {r.stage,-10} play {r.playMin:F1} 分  python {r.python} {r.pythonMin:F1} 分  {r.reason}");
        try
        {
            if (!string.IsNullOrEmpty(st.batchDir))
                File.WriteAllText(Path.Combine(st.batchDir, $"batch_summary_{DateTime.Now:yyyyMMdd_HHmmss}.json"), JsonUtility.ToJson(st, true), new UTF8Encoding(false));
        }
        catch { }
    }

    // ================= 小物 =================

    static JobRecord Cur() => st.records.Count > 0 ? st.records[st.records.Count - 1] : new JobRecord();

    static BatchJob.Result ReadResult()
    {
        try
        {
            if (!File.Exists(BatchJob.ResultPath)) return null;
            var r = JsonUtility.FromJson<BatchJob.Result>(File.ReadAllText(BatchJob.ResultPath, Encoding.UTF8));
            return r != null && r.jobId == st.curJobId ? r : null;
        }
        catch { return null; }
    }

    /// 途中まで書かれた scene フォルダを <root>/_failed/<jobId>/ へ移す(消さない)
    static string MoveFailed(JobRecord rec)
    {
        string dst = Path.Combine(st.rootAbs, "_failed", rec.jobId);
        bool moved = false;
        foreach (int i in new[] { rec.sceneIndex, rec.sceneIndex + 1 })
        {
            string src = Path.Combine(st.rootAbs, $"scene_{i:D4}");
            if (!Directory.Exists(src)) continue;
            Directory.CreateDirectory(dst);
            Directory.Move(src, Path.Combine(dst, $"scene_{i:D4}"));
            moved = true;
        }
        return moved ? dst : "";
    }

    static string FindClash(string root, int sceneIndex, int pairIndex)
    {
        foreach (var p in new[] { Path.Combine(root, $"scene_{sceneIndex:D4}"), Path.Combine(root, $"scene_{sceneIndex + 1:D4}"), Path.Combine(root, "pairs", $"pair_{pairIndex:D4}") })
            if (Directory.Exists(p)) return p;
        return null;
    }

    static bool ProcessAlive(int pid)
    {
        if (pid <= 0) return false;
        try { var p = Process.GetProcessById(pid); return !p.HasExited; }
        catch { return false; }
    }

    static string Git(string args)
    {
        var psi = new ProcessStartInfo("git", args)
        {
            WorkingDirectory = ProjectDir, UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true
        };
        psi.EnvironmentVariables["GIT_TERMINAL_PROMPT"] = "0";
        using (var p = Process.Start(psi))
        {
            var o = p.StandardOutput.ReadToEndAsync();   // 標準出力と標準エラーを同時に読む(デッドロック防止)
            var e = p.StandardError.ReadToEndAsync();
            if (!p.WaitForExit(20000)) { try { p.Kill(); } catch { } return "(git timeout)"; }
            return o.Result;
        }
    }

    static void Log(string msg)
    {
        string line = $"[{Local}] {msg}";
        Debug.Log("[BatchRunner] " + msg);
        try
        {
            if (st != null && !string.IsNullOrEmpty(st.batchDir))
            {
                Directory.CreateDirectory(st.batchDir);
                File.AppendAllText(Path.Combine(st.batchDir, "batch_log.txt"), line + "\n", new UTF8Encoding(false));
            }
        }
        catch { }
    }

    static State LoadState()
    {
        try
        {
            if (File.Exists(StatePath)) return JsonUtility.FromJson<State>(File.ReadAllText(StatePath, Encoding.UTF8));
        }
        catch (Exception e) { Debug.LogWarning("[BatchRunner] state.json を読めない: " + e.Message); }
        return new State();
    }

    static void Save()
    {
        try
        {
            Directory.CreateDirectory(BatchJob.Dir);
            File.WriteAllText(StatePath, JsonUtility.ToJson(st, true), new UTF8Encoding(false));
        }
        catch (Exception e) { Debug.LogWarning("[BatchRunner] state.json を書けない: " + e.Message); }
    }
}
