using System;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// P6: バッチ実行(Editor の BatchRunner)と EpisodeRecorder の受け渡し。
/// - BatchRunner が Play の前に job.json を書く → EpisodeRecorder が Start で読んで消す(1 回だけ効く)
/// - EpisodeRecorder が終わったら result.json を書く → BatchRunner が見て Play を止める
/// 値は Play 中のインスタンスにだけ入るので、シーンファイルは変わらない。
/// 置き場所は &lt;project&gt;/Temp/p6_batch/(git に入らない、Unity を閉じると消える)
/// </summary>
public static class BatchJob
{
    [Serializable]
    public class Job
    {
        public string jobId = "";
        public int seed;
        public int sceneIndex;
        public int pairIndex;
        public string outputRoot = "";   // 絶対パス。空なら EpisodeRecorder の既定(Episodes)
    }

    [Serializable]
    public class Result
    {
        public string jobId = "";
        public bool ok;
        public string stage = "";        // done / generate / kinematics / fk / plan / start
        public string reason = "";
        public int seed;
        public int sceneIndex;
        public int pairIndex;
        public int candidatesPlanned;
        public int candidatesExecuted;
        public string finishedLocal = "";
    }

    public static string Dir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Temp", "p6_batch"));
    public static string JobPath => Path.Combine(Dir, "job.json");
    public static string ResultPath => Path.Combine(Dir, "result.json");

    /// job.json があれば読んで消す。無ければ null
    public static Job TryConsume()
    {
        try
        {
            if (!File.Exists(JobPath)) return null;
            var job = JsonUtility.FromJson<Job>(File.ReadAllText(JobPath, Encoding.UTF8));
            File.Delete(JobPath);
            return job;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[BatchJob] job.json を読めない: {e.Message}");
            return null;
        }
    }

    public static void WriteResult(Result r)
    {
        try
        {
            Directory.CreateDirectory(Dir);
            r.finishedLocal = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            string tmp = ResultPath + ".tmp";
            File.WriteAllText(tmp, JsonUtility.ToJson(r, true), new UTF8Encoding(false));
            if (File.Exists(ResultPath)) File.Delete(ResultPath);
            File.Move(tmp, ResultPath);   // 書きかけを読まれないように、書き終わってから名前を変える
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[BatchJob] result.json を書けない: {e.Message}");
        }
    }
}
