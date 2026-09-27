using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.LabelManagement;

/// <summary>
/// TODO5: 1 episode 分のデータを作る。
/// 現在の実装範囲: 2. Initial scene の保存
///   物体を落ち着かせる → 固定 → ロボットあり / robot-free を撮影
///   → camera.json・scene_initial.json を書く → SOLO から画像をコピー → 固定を解除
/// 出力: &lt;project&gt;/Episodes/scene_XXXX/
/// 前提: PerceptionCamera の Capture Trigger Mode = Manual。RobotFreeCapture は無効にしておく。
/// </summary>
public class EpisodeRecorder : MonoBehaviour
{
    [Header("References (空なら自動で探す)")]
    public PerceptionCamera perceptionCamera;
    public CameraExporter cameraExporter;
    public Transform robotRoot;      // ur3_with_gripper
    public Transform objectsRoot;    // Objects

    [Header("Scene")]
    public int sceneIndex = 0;
    public int seed = 0;
    [Tooltip("空なら <project>/Episodes")]
    public string outputRoot = "";

    [Header("Settle (物体を落ち着かせる)")]
    public int settleFixedSteps = 100;
    public float settleMaxDisplacement = 0.001f;  // m(配置位置からのズレ)
    public float settleMaxSpeed = 0.001f;         // m/s

    [Header("Capture")]
    public int framesBeforeStart = 30;            // RobotFreeCapture と同じ
    public int hideSettleFrames = 1;              // RobotFreeCapture と同じ

    [Header("SOLO")]
    [Tooltip("空なら Application.persistentDataPath")]
    public string soloRoot = "";
    public float fileWaitTimeoutSec = 30f;

    Renderer[] robotRenderers;
    bool[] originalForceOff;
    Rigidbody[] bodies;
    Vector3[] authoredPositions;
    int captureIndex = 0;
    DateTime runStartUtc;

    void Start()
    {
        if (perceptionCamera == null) perceptionCamera = FindFirstObjectByType<PerceptionCamera>();
        if (cameraExporter == null && perceptionCamera != null) cameraExporter = perceptionCamera.GetComponent<CameraExporter>();
        if (robotRoot == null) { var go = GameObject.Find("ur3_with_gripper"); if (go != null) robotRoot = go.transform; }
        if (objectsRoot == null) { var go = GameObject.Find("Objects"); if (go != null) objectsRoot = go.transform; }

        if (perceptionCamera == null || cameraExporter == null || robotRoot == null || objectsRoot == null)
        {
            Debug.LogError("[EpisodeRecorder] perceptionCamera / cameraExporter / robotRoot / objectsRoot のどれかが見つからない");
            enabled = false;
            return;
        }

        var rfc = FindFirstObjectByType<RobotFreeCapture>();
        if (rfc != null && rfc.enabled)
        {
            Debug.LogError("[EpisodeRecorder] RobotFreeCapture が有効。撮影の step 番号が混ざるので無効にして");
            enabled = false;
            return;
        }

        robotRenderers = robotRoot.GetComponentsInChildren<Renderer>(true);
        originalForceOff = new bool[robotRenderers.Length];
        for (int i = 0; i < robotRenderers.Length; i++)
            originalForceOff[i] = robotRenderers[i].forceRenderingOff;

        // 配置した位置(物理が1ステップも回る前)を記録しておく
        bodies = objectsRoot.GetComponentsInChildren<Rigidbody>();
        authoredPositions = bodies.Select(b => b.position).ToArray();

        runStartUtc = DateTime.UtcNow;
        StartCoroutine(Run());
    }

    IEnumerator Run()
    {
        for (int i = 0; i < framesBeforeStart; i++) yield return null;

        string root = string.IsNullOrEmpty(outputRoot)
            ? Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Episodes"))
            : outputRoot;
        string sceneId = $"scene_{sceneIndex:D4}";
        string sceneDir = Path.Combine(root, sceneId);
        if (Directory.Exists(sceneDir))
            Debug.LogWarning($"[EpisodeRecorder] {sceneDir} は既にある。同じ名前のファイルは上書きする");
        Directory.CreateDirectory(sceneDir);

        // ---- 1) 物体を落ち着かせる ----
        for (int i = 0; i < settleFixedSteps; i++) yield return new WaitForFixedUpdate();
        float maxDisp = 0f, maxSpeed = 0f;
        for (int i = 0; i < bodies.Length; i++)
        {
            maxDisp = Mathf.Max(maxDisp, Vector3.Distance(bodies[i].position, authoredPositions[i]));
            maxSpeed = Mathf.Max(maxSpeed, bodies[i].linearVelocity.magnitude);
        }
        bool settled = maxDisp <= settleMaxDisplacement && maxSpeed <= settleMaxSpeed;
        if (!settled)
            Debug.LogWarning($"[EpisodeRecorder] 物体が落ち着いていない: max displacement={maxDisp:F6} m, max speed={maxSpeed:F6} m/s");

        // ---- 2) 撮影中は固定する ----
        var origKinematic = new bool[bodies.Length];
        var origMode = new CollisionDetectionMode[bodies.Length];
        for (int i = 0; i < bodies.Length; i++)
        {
            origKinematic[i] = bodies[i].isKinematic;
            origMode[i] = bodies[i].collisionDetectionMode;
            bodies[i].collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
            bodies[i].isKinematic = true;
        }
        string objectsJson = BuildObjectsJson();
        string robotJson = BuildRobotJson();

        // ---- 3) 撮影: ロボットあり → robot-free ----
        int stepWithRobot = captureIndex;
        yield return CaptureOne();
        SetRobotVisible(false);
        for (int i = 0; i < hideSettleFrames; i++) yield return null;
        int stepRobotFree = captureIndex;
        yield return CaptureOne();
        SetRobotVisible(true);

        // ---- 4) カメラ行列 ----
        string camPath = cameraExporter.ExportToFile(Path.Combine(sceneDir, "camera.json"));

        // ---- 5) SOLO から画像をコピー ----
        string soloDir = null;
        yield return FindSoloDir(d => soloDir = d);
        bool okWith = false, okFree = false;
        if (soloDir != null)
        {
            yield return CopyStep(soloDir, stepWithRobot, Path.Combine(sceneDir, "initial", "with_robot"), r => okWith = r);
            yield return CopyStep(soloDir, stepRobotFree, Path.Combine(sceneDir, "initial", "robot_free"), r => okFree = r);
            string defs = Path.Combine(soloDir, "annotation_definitions.json");
            if (File.Exists(defs)) File.Copy(defs, Path.Combine(sceneDir, "solo_annotation_definitions.json"), true);
        }

        // ---- 6) scene_initial.json ----
        WriteSceneJson(sceneDir, sceneId, settled, maxDisp, maxSpeed, soloDir,
                       stepWithRobot, stepRobotFree, okWith, okFree, camPath != null, objectsJson, robotJson);

        // ---- 7) 固定を解除(この後の実行で物体が動けるように) ----
        for (int i = 0; i < bodies.Length; i++)
        {
            bodies[i].isKinematic = origKinematic[i];
            bodies[i].collisionDetectionMode = origMode[i];
        }

        if (settled && okWith && okFree && camPath != null)
            Debug.Log($"[EpisodeRecorder] Initial scene 保存完了: {sceneDir}");
        else
            Debug.LogWarning($"[EpisodeRecorder] Initial scene 保存に問題あり: settled={settled}, with_robot={okWith}, robot_free={okFree}, camera={camPath != null} ({sceneDir})");
    }

    // ---------- capture ----------

    IEnumerator CaptureOne()
    {
        perceptionCamera.RequestCapture();
        captureIndex++;
        yield return null;
    }

    void SetRobotVisible(bool visible)
    {
        for (int i = 0; i < robotRenderers.Length; i++)
        {
            if (robotRenderers[i] == null) continue;
            robotRenderers[i].forceRenderingOff = visible ? originalForceOff[i] : true;
        }
    }

    // ---------- SOLO ----------

    IEnumerator FindSoloDir(Action<string> done)
    {
        string baseDir = string.IsNullOrEmpty(soloRoot) ? Application.persistentDataPath : soloRoot;
        float t0 = Time.realtimeSinceStartup;
        while (Time.realtimeSinceStartup - t0 < fileWaitTimeoutSec)
        {
            if (Directory.Exists(baseDir))
            {
                var dir = Directory.GetDirectories(baseDir, "solo*")
                    .Where(d => Directory.GetCreationTimeUtc(d) >= runStartUtc.AddSeconds(-5))
                    .OrderByDescending(d => Directory.GetCreationTimeUtc(d))
                    .FirstOrDefault();
                if (dir != null) { done(dir); yield break; }
            }
            yield return new WaitForSecondsRealtime(0.2f);
        }
        Debug.LogWarning($"[EpisodeRecorder] この実行の SOLO フォルダが見つからない({baseDir})");
        done(null);
    }

    IEnumerator CopyStep(string soloDir, int step, string destDir, Action<bool> done)
    {
        string seq = Path.Combine(soloDir, "sequence.0");
        string prefix = $"step{step}.";
        string[] required = { "rgb.png", "semantic.png", "depth.exr" };
        var lastSizes = new Dictionary<string, long>();
        float t0 = Time.realtimeSinceStartup;

        while (Time.realtimeSinceStartup - t0 < fileWaitTimeoutSec)
        {
            if (Directory.Exists(seq))
            {
                var files = Directory.GetFiles(seq, prefix + "*");
                var sizes = files.ToDictionary(f => MapName(Path.GetFileName(f), prefix), f => new FileInfo(f).Length);
                bool allThere = required.All(r => sizes.ContainsKey(r) && sizes[r] > 0);
                bool stable = allThere && sizes.Count == lastSizes.Count &&
                              sizes.All(kv => lastSizes.TryGetValue(kv.Key, out var s) && s == kv.Value);
                if (stable)
                {
                    Directory.CreateDirectory(destDir);
                    foreach (var f in files)
                        File.Copy(f, Path.Combine(destDir, MapName(Path.GetFileName(f), prefix)), true);
                    done(true);
                    yield break;
                }
                lastSizes = sizes;
            }
            yield return new WaitForSecondsRealtime(0.3f);
        }
        Debug.LogWarning($"[EpisodeRecorder] step{step} のファイルが揃わなかった({seq})");
        done(false);
    }

    static string MapName(string fileName, string prefix)
    {
        string rest = fileName.Substring(prefix.Length);   // 例: "camera.png"
        string low = rest.ToLowerInvariant();
        if (low == "camera.png") return "rgb.png";
        if (low.Contains("semantic")) return "semantic.png";
        if (low.Contains("instance")) return "instance.png";
        if (low.EndsWith(".exr")) return "depth.exr";
        if (low == "frame_data.json") return "frame_data.json";
        return rest;
    }

    // ---------- JSON ----------

    string BuildObjectsJson()
    {
        var items = new List<string>();
        foreach (var b in bodies)
        {
            var col = b.GetComponent<Collider>();
            var lab = b.GetComponent<Labeling>();
            string labels = lab != null ? "[" + string.Join(", ", lab.labels.Select(Q)) + "]" : "[]";
            var mat = col != null ? col.sharedMaterial : null;
            bool defaultMat = mat == null;
            float sf = defaultMat ? 0.6f : mat.staticFriction;
            float df = defaultMat ? 0.6f : mat.dynamicFriction;
            float bo = defaultMat ? 0f : mat.bounciness;
            Bounds bw = col != null ? col.bounds : new Bounds(b.position, Vector3.zero);

            items.Add("    {" +
                $"\"name\": {Q(b.name)}, " +
                $"\"path\": {Q(PathOf(b.transform))}, " +
                $"\"labels\": {labels}, " +
                $"\"position_world\": {Vec(b.position)}, " +
                $"\"rotation_world_xyzw\": {Quat(b.rotation)}, " +
                $"\"scale\": {Vec(b.transform.lossyScale)}, " +
                $"\"bounds_size_world\": {Vec(bw.size)}, " +
                $"\"collider\": {Q(col != null ? col.GetType().Name : "none")}, " +
                $"\"mass\": {N(b.mass)}, " +
                $"\"physics_material\": {Q(defaultMat ? "default" : mat.name)}, " +
                $"\"static_friction\": {N(sf)}, \"dynamic_friction\": {N(df)}, \"bounciness\": {N(bo)}" +
                "}");
        }
        return "[\n" + string.Join(",\n", items) + "\n  ]";
    }

    string BuildRobotJson()
    {
        var joints = new List<string>();
        foreach (var ab in robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (!ab.isRoot && ab.dofCount > 0)
                joints.Add($"      {{\"name\": {Q(ab.name)}, \"position\": {N(ab.jointPosition[0])}}}");
        return "{\n" +
               $"    \"name\": {Q(robotRoot.name)},\n" +
               $"    \"root_position_world\": {Vec(robotRoot.position)},\n" +
               "    \"joints\": [\n" + string.Join(",\n", joints) + "\n    ]\n  }";
    }

    void WriteSceneJson(string sceneDir, string sceneId, bool settled, float maxDisp, float maxSpeed,
                        string soloDir, int stepWith, int stepFree, bool okWith, bool okFree, bool okCam,
                        string objectsJson, string robotJson)
    {
        var f = new List<string>();
        f.Add(KV("schema", Q("scene_initial_v1")));
        f.Add(KV("scene_id", Q(sceneId)));
        f.Add(KV("scene_index", sceneIndex.ToString(CultureInfo.InvariantCulture)));
        f.Add(KV("seed", seed.ToString(CultureInfo.InvariantCulture)));
        f.Add(KV("created_at", Q(DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture))));
        f.Add(KV("unity_version", Q(Application.unityVersion)));
        f.Add(KV("world_frame", Q("Unity world: left-handed, Y-up, meters")));
        f.Add(KV("units", "{\"length\": \"m\", \"joint_revolute\": \"rad\", \"joint_prismatic\": \"m\", \"mass\": \"kg\", \"time\": \"s\"}"));
        f.Add(KV("physics", "{" +
            $"\"fixed_delta_time\": {N(Time.fixedDeltaTime)}, " +
            $"\"gravity\": {Vec(Physics.gravity)}, " +
            $"\"default_contact_offset\": {N(Physics.defaultContactOffset)}, " +
            $"\"solver_iterations\": {Physics.defaultSolverIterations}, " +
            $"\"solver_velocity_iterations\": {Physics.defaultSolverVelocityIterations}, " +
            $"\"bounce_threshold\": {N(Physics.bounceThreshold)}}}"));
        f.Add(KV("settle", "{" +
            $"\"fixed_steps\": {settleFixedSteps}, " +
            $"\"max_displacement_m\": {N(maxDisp)}, \"max_speed_mps\": {N(maxSpeed)}, " +
            $"\"threshold_displacement_m\": {N(settleMaxDisplacement)}, \"threshold_speed_mps\": {N(settleMaxSpeed)}, " +
            $"\"settled\": {(settled ? "true" : "false")}}}"));
        f.Add(KV("robot", robotJson));
        f.Add(KV("objects", objectsJson));
        f.Add(KV("capture", "{" +
            $"\"solo_dir\": {(soloDir != null ? Q(soloDir) : "null")}, " +
            $"\"step_with_robot\": {stepWith}, \"step_robot_free\": {stepFree}, " +
            $"\"copied_with_robot\": {(okWith ? "true" : "false")}, \"copied_robot_free\": {(okFree ? "true" : "false")}}}"));
        f.Add(KV("files", "{\n" +
            $"    \"camera\": {(okCam ? Q("camera.json") : "null")},\n" +
            "    \"robot_free\": {\"rgb\": \"initial/robot_free/rgb.png\", \"depth\": \"initial/robot_free/depth.exr\", " +
            "\"semantic\": \"initial/robot_free/semantic.png\", \"instance\": \"initial/robot_free/instance.png\"},\n" +
            "    \"with_robot\": {\"rgb\": \"initial/with_robot/rgb.png\", \"depth\": \"initial/with_robot/depth.exr\", " +
            "\"semantic\": \"initial/with_robot/semantic.png\", \"instance\": \"initial/with_robot/instance.png\"}\n  }"));

        string json = "{\n" + string.Join(",\n", f) + "\n}\n";
        File.WriteAllText(Path.Combine(sceneDir, "scene_initial.json"), json, new UTF8Encoding(false));
    }

    static string PathOf(Transform t)
    {
        var parts = new List<string>();
        for (var c = t; c != null; c = c.parent) parts.Add(c.name);
        parts.Reverse();
        return string.Join("/", parts);
    }

    static string N(double d) => d.ToString("G10", CultureInfo.InvariantCulture);
    static string Q(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    static string KV(string k, string v) => $"  \"{k}\": {v}";
    static string Vec(Vector3 v) => $"[{N(v.x)}, {N(v.y)}, {N(v.z)}]";
    static string Quat(Quaternion q) => $"[{N(q.x)}, {N(q.y)}, {N(q.z)}, {N(q.w)}]";
}
