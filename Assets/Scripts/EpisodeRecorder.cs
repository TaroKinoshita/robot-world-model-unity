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
/// 現在の実装範囲:
///   2. Initial scene の保存(物体を落ち着かせる → 固定 → ロボットあり / robot-free を撮影 → JSON と画像を保存)
///   3. 候補軌道の作成(FK の検証 → 候補ごとに IK で軌道を計画 → planned_trajectory を保存)
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

    [Header("Robot (UR5e に替えるときはここを変える)")]
    public string toolLinkName = "tool0";
    public string[] armJointNames = {
        "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link" };
    public string fingerPadA = "left_inner_finger_pad";
    public string fingerPadB = "right_inner_finger_pad";
    [Tooltip("FK を実際のロボットの姿勢と比べて検証する(ロボットを一時的に高い姿勢へ動かす)")]
    public bool validateKinematics = true;
    public float fkToleranceM = 0.001f;

    [Header("Candidates")]
    public string targetObjectName = "Target";
    [Tooltip("candidateSpecs の先頭から何本使うか")]
    public int numCandidates = 1;
    public List<CandidateSpec> candidateSpecs = new List<CandidateSpec>
    {
        new CandidateSpec { note = "push away from robot (+Z)", pushAngleDeg = 0f, pushLength = 0.10f },
        new CandidateSpec { note = "push toward SecondObject", pushAngleDeg = 130f, pushLength = 0.10f },
    };
    public PlannerSettings planner = new PlannerSettings();
    public ActionRasterSettings actions = new ActionRasterSettings();
    [Header("Execution (5. 実行)")]
    public DriveSettings drives = new DriveSettings();
    [Tooltip("軌道の最後のあと、物体が止まるまで記録を続けるステップ数")]
    public int postSettleSteps = 50;
    [Tooltip("リセット後、物体を固定したまま待つステップ数")]
    public int resetHoldSteps = 3;
    [Header("Final / labels (6. 最終結果)")]
    [Tooltip("ゴール領域の中心(world、XZ だけ使う)。仮の値: ターゲットを -X に 10cm 動かした位置")]
    public Vector3 goalCenter = new Vector3(-0.10f, 0f, 0.30f);
    public float goalRadius = 0.03f;
    public float fallDropThreshold = 0.02f;
    public Color32 targetMaskColor = new Color32(255, 0, 0, 255);
    string runSoloDir;

    Renderer[] robotRenderers;
    bool[] originalForceOff;
    Rigidbody[] bodies;
    Vector3[] authoredPositions;
    ArmKinematics kin;
    string kinError;
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

        // 運動学は初期姿勢(指が開いた状態)で作る
        EnsureContactRecorders();   // 物理が1回も回る前に付ける
        kin = ArmKinematics.Build(robotRoot, toolLinkName, armJointNames, fingerPadA, fingerPadB, out kinError);
        if (kin == null) Debug.LogError($"[EpisodeRecorder] 運動学を作れない: {kinError}");

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

        // ================= 2. Initial scene =================

        // ---- 物体を落ち着かせる ----
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

        // ---- 撮影と計画の間は固定する ----
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
        settledPos = bodies.Select(b => b.position).ToArray();
        settledRot = bodies.Select(b => b.rotation).ToArray();
        savedKinematic = origKinematic;
        savedMode = origMode;
        initialDofPositions = CaptureAllDofs();
        float[] qInit = kin != null ? ReadArmJoints() : null;

        // ---- 撮影: ロボットあり → robot-free ----
        int stepWithRobot = captureIndex;
        yield return CaptureOne();
        SetRobotVisible(false);
        for (int i = 0; i < hideSettleFrames; i++) yield return null;
        int stepRobotFree = captureIndex;
        yield return CaptureOne();
        SetRobotVisible(true);

        // ---- カメラ行列 ----
        string camPath = cameraExporter.ExportToFile(Path.Combine(sceneDir, "camera.json"));

        // ---- SOLO から画像をコピー ----
        string soloDir = null;
        yield return FindSoloDir(d => soloDir = d);
        runSoloDir = soloDir;
        bool okWith = false, okFree = false;
        if (soloDir != null)
        {
            yield return CopyStep(soloDir, stepWithRobot, Path.Combine(sceneDir, "initial", "with_robot"), r => okWith = r);
            yield return CopyStep(soloDir, stepRobotFree, Path.Combine(sceneDir, "initial", "robot_free"), r => okFree = r);
            string defs = Path.Combine(soloDir, "annotation_definitions.json");
            if (File.Exists(defs)) File.Copy(defs, Path.Combine(sceneDir, "solo_annotation_definitions.json"), true);
        }

        WriteSceneJson(sceneDir, sceneId, settled, maxDisp, maxSpeed, soloDir,
                       stepWithRobot, stepRobotFree, okWith, okFree, camPath != null, objectsJson, robotJson);

        bool initialOk = settled && okWith && okFree && camPath != null;
        if (initialOk) Debug.Log($"[EpisodeRecorder] Initial scene 保存完了: {sceneDir}");
        else Debug.LogWarning($"[EpisodeRecorder] Initial scene 保存に問題あり: settled={settled}, with_robot={okWith}, robot_free={okFree}, camera={camPath != null}");

        // ================= 3. 候補軌道 =================

        List<PlanResult> plans = null;
        if (kin == null)
        {
            Debug.LogError($"[EpisodeRecorder] 運動学が無いので候補軌道を作れない: {kinError}");
        }
        else
        {
            // ---- FK の検証(ロボットを高い姿勢へ動かして、計算と実際を比べる) ----
            float fkErrPlus = -1f, fkErrMinus = -1f;
            bool fkOk = true;
            if (validateKinematics)
            {
                yield return ValidateFK(qInit, (ep, em) => { fkErrPlus = ep; fkErrMinus = em; });
                kin.JointSign = fkErrPlus <= fkErrMinus ? 1f : -1f;
                float best = Mathf.Min(fkErrPlus, fkErrMinus);
                fkOk = best <= fkToleranceM;
                if (fkOk) Debug.Log($"[EpisodeRecorder] FK 検証 OK: 最大誤差 {best * 1000f:F3} mm(joint sign {kin.JointSign:+0;-0})");
                else Debug.LogError($"[EpisodeRecorder] FK 検証 NG: 最大誤差 +:{fkErrPlus * 1000f:F2} mm / -:{fkErrMinus * 1000f:F2} mm");
            }

            if (fkOk) plans = PlanCandidates(sceneDir, sceneId, qInit, fkErrPlus, fkErrMinus);
        }

        // ================= 5. 実行(物体は固定したまま始める) =================
        if (plans != null && plans.Count > 0)
        {
            Debug.Log("[EpisodeRecorder] 5. ドライブ設定");
            ApplyDrives();
            yield return new WaitForFixedUpdate();
            foreach (var p in plans) yield return ExecuteCandidate(p);
        }
        else
        {
            for (int i = 0; i < bodies.Length; i++) bodies[i].isKinematic = origKinematic[i];
        }

        Debug.Log("[EpisodeRecorder] 完了");
    }

    // ---------- 3. candidates ----------

    List<PlanResult> PlanCandidates(string sceneDir, string sceneId, float[] qInit, float fkErrPlus, float fkErrMinus)
    {
        var target = bodies.FirstOrDefault(b => b.name == targetObjectName);
        if (target == null) { Debug.LogError($"[EpisodeRecorder] ターゲット '{targetObjectName}' が無い"); return null; }
        var results = new List<PlanResult>();
        var col = target.GetComponent<Collider>();
        Vector3 center = col != null ? col.bounds.center : target.position;
        Vector3 half = col != null ? col.bounds.extents : Vector3.one * 0.04f;

        int n = Mathf.Clamp(numCandidates, 0, candidateSpecs.Count);
        var index = new List<string>();
        for (int c = 0; c < n; c++)
        {
            string cid = $"c{c:D3}";
            var spec = candidateSpecs[c];
            var tr = TrajectoryPlanner.Plan(kin, qInit, center, half, spec, planner, Time.fixedDeltaTime);
            string dir = Path.Combine(sceneDir, "candidates", cid);
            Directory.CreateDirectory(dir);
            WritePlannedJson(Path.Combine(dir, "planned_trajectory.json"), sceneId, cid, c, spec, tr, center, half);
            WritePlannedCsv(Path.Combine(dir, "planned_trajectory.csv"), tr);
            if (tr.ok) ActionRepresentations.Write(kin, tr, perceptionCamera.GetComponent<Camera>(), dir, actions, sceneId, cid);
            if (tr.ok) results.Add(new PlanResult { cid = cid, dir = dir, tr = tr, sceneId = sceneId });

            index.Add("    {" +
                $"\"candidate_id\": {Q(cid)}, \"candidate_index\": {c}, \"note\": {Q(spec.note)}, " +
                $"\"push_angle_deg\": {N(spec.pushAngleDeg)}, \"push_length_m\": {N(spec.pushLength)}, " +
                $"\"planned_ok\": {(tr.ok ? "true" : "false")}, \"num_steps\": {tr.q.Count}, " +
                $"\"planned_trajectory\": {Q($"candidates/{cid}/planned_trajectory.json")}" +
                (tr.ok ? "" : $", \"error\": {Q(tr.error ?? "")}") + "}");

            if (tr.ok)
                Debug.Log($"[EpisodeRecorder] {cid} 計画 OK: {tr.q.Count} steps ({tr.q.Count * tr.dt:F2} s), " +
                          $"IK 最大誤差 {tr.maxIkPosErr * 1000f:F3} mm / {tr.maxIkAngErr:F4} rad, 1 step の最大関節変化 {tr.maxJointStep * Mathf.Rad2Deg:F2}°, " +
                          $"移動中の TCP 最低高さ {tr.minTcpHeightTransfer:F3} m");
            else
                Debug.LogError($"[EpisodeRecorder] {cid} 計画 NG: {tr.error}");
        }

        string json = "{\n" +
            $"  \"schema\": \"candidates_v1\",\n" +
            $"  \"scene_id\": {Q(sceneId)},\n" +
            $"  \"num_candidates\": {n},\n" +
            $"  \"fk_validation\": {{\"enabled\": {(validateKinematics ? "true" : "false")}, \"max_err_sign_plus_m\": {N(fkErrPlus)}, \"max_err_sign_minus_m\": {N(fkErrMinus)}, \"joint_sign\": {N(kin.JointSign)}}},\n" +
            "  \"candidates\": [\n" + string.Join(",\n", index) + "\n  ]\n}\n";
        File.WriteAllText(Path.Combine(sceneDir, "candidates.json"), json, new UTF8Encoding(false));
        return results;
    }

    IEnumerator ValidateFK(float[] qInit, Action<float, float> done)
    {
        // 物体に届かない高い姿勢(rad)
        float[][] tests =
        {
            new[] { 0.6f, 0.3f, -0.4f, 0.2f, 0.5f, 0.3f },
            new[] { -1.0f, -0.3f, 0.5f, -0.6f, -0.4f, 1.0f },
            new[] { 2.2f, 0.2f, 0.3f, 0.4f, -0.8f, -0.5f },
        };
        float errPlus = 0f, errMinus = 0f;
        float savedSign = kin.JointSign;
        foreach (var q in tests)
        {
            Teleport(q);
            for (int i = 0; i < 3; i++) yield return new WaitForFixedUpdate();
            float[] qActual = ReadArmJoints();
            Vector3 actual = kin.Tool.TransformPoint(kin.TcpLocal);

            kin.JointSign = 1f;
            kin.TcpPose(qActual, out var pPlus, out _);
            kin.JointSign = -1f;
            kin.TcpPose(qActual, out var pMinus, out _);
            errPlus = Mathf.Max(errPlus, Vector3.Distance(pPlus, actual));
            errMinus = Mathf.Max(errMinus, Vector3.Distance(pMinus, actual));
        }
        kin.JointSign = savedSign;
        Teleport(qInit);
        for (int i = 0; i < 5; i++) yield return new WaitForFixedUpdate();
        done(errPlus, errMinus);
    }

    void Teleport(float[] q)
    {
        for (int j = 0; j < kin.Dof; j++)
        {
            var ab = kin.JointBodies[j];
            var d = ab.xDrive;
            d.target = q[j] * Mathf.Rad2Deg;
            ab.xDrive = d;
            ab.jointPosition = new ArticulationReducedSpace(q[j]);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
        }
    }

    float[] ReadArmJoints()
    {
        var q = new float[kin.Dof];
        for (int j = 0; j < kin.Dof; j++) q[j] = kin.JointBodies[j].jointPosition[0];
        return q;
    }

    void WritePlannedJson(string path, string sceneId, string cid, int cIndex, CandidateSpec spec, PlannedTrajectory tr,
                          Vector3 center, Vector3 half)
    {
        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"schema\": \"planned_trajectory_v1\",\n");
        sb.Append($"  \"scene_id\": {Q(sceneId)},\n");
        sb.Append($"  \"candidate_id\": {Q(cid)},\n");
        sb.Append($"  \"candidate_index\": {cIndex},\n");
        sb.Append($"  \"planned_ok\": {(tr.ok ? "true" : "false")},\n");
        if (!tr.ok) sb.Append($"  \"error\": {Q(tr.error ?? "")},\n");
        sb.Append($"  \"spec\": {{\"note\": {Q(spec.note)}, \"push_angle_deg\": {N(spec.pushAngleDeg)}, \"push_length_m\": {N(spec.pushLength)}}},\n");
        sb.Append($"  \"planner\": {JsonUtility.ToJson(planner)},\n");
        sb.Append($"  \"target\": {{\"name\": {Q(targetObjectName)}, \"center_world\": {Vec(center)}, \"half_extents_world\": {Vec(half)}}},\n");
        sb.Append($"  \"push_direction_world\": {Vec(tr.pushDir)},\n");
        sb.Append("  \"waypoints\": {" + string.Join(", ", tr.waypointNames.Select((w, i) => $"{Q(w)}: {Vec(tr.waypoints[i])}")) + "},\n");
        sb.Append($"  \"robot\": {{\"name\": {Q(robotRoot.name)}, \"tool_link\": {Q(toolLinkName)}, \"joint_sign\": {N(kin.JointSign)}}},\n");
        sb.Append($"  \"tcp_definition\": {{\"description\": \"midpoint of the two finger pads\", \"pads\": [{Q(fingerPadA)}, {Q(fingerPadB)}], " +
                  $"\"tcp_in_tool_frame\": {Vec(kin.TcpLocal)}, \"approach_in_tool_frame\": {Vec(kin.ApproachLocal)}, " +
                  $"\"opening_in_tool_frame\": {Vec(kin.OpeningLocal)}, \"finger_half_span_m\": {N(kin.FingerHalfSpan)}}},\n");
        sb.Append("  \"gripper\": \"open; finger joints are not commanded\",\n");
        sb.Append("  \"frames\": {\"world\": \"Unity world: left-handed, Y-up, meters\", \"joints\": \"Unity ArticulationBody jointPosition, rad\"},\n");
        sb.Append($"  \"quality\": {{\"max_ik_pos_err_m\": {N(tr.maxIkPosErr)}, \"max_ik_ang_err_rad\": {N(tr.maxIkAngErr)}, " +
                  $"\"max_joint_step_rad\": {N(tr.maxJointStep)}, \"min_tcp_height_transfer_m\": {N(tr.q.Count > 0 ? tr.minTcpHeightTransfer : 0f)}}},\n");
        sb.Append($"  \"dt\": {N(tr.dt)},\n");
        sb.Append($"  \"num_steps\": {tr.q.Count},\n");
        sb.Append("  \"joint_names\": [" + string.Join(", ", kin.JointNames.Select(Q)) + "],\n");
        sb.Append("  \"steps\": {\n");
        sb.Append("    \"t\": [" + string.Join(", ", tr.t.Select(v => N(v))) + "],\n");
        sb.Append("    \"phase\": [" + string.Join(", ", tr.phase.Select(Q)) + "],\n");
        sb.Append("    \"q\": [" + string.Join(", ", tr.q.Select(q => "[" + string.Join(", ", q.Select(v => N(v))) + "]")) + "],\n");
        sb.Append("    \"tcp_position\": [" + string.Join(", ", tr.tcp.Select(Vec)) + "],\n");
        sb.Append("    \"tool_rotation_xyzw\": [" + string.Join(", ", tr.toolRot.Select(Quat)) + "]\n");
        sb.Append("  }\n}\n");
        File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
    }

    void WritePlannedCsv(string path, PlannedTrajectory tr)
    {
        var sb = new StringBuilder();
        sb.Append("step,t,phase," + string.Join(",", kin.JointNames) + ",tcp_x,tcp_y,tcp_z\n");
        for (int i = 0; i < tr.q.Count; i++)
        {
            sb.Append(i).Append(',').Append(N(tr.t[i])).Append(',').Append(tr.phase[i]);
            foreach (var v in tr.q[i]) sb.Append(',').Append(N(v));
            sb.Append(',').Append(N(tr.tcp[i].x)).Append(',').Append(N(tr.tcp[i].y)).Append(',').Append(N(tr.tcp[i].z)).Append('\n');
        }
        File.WriteAllText(path, sb.ToString(), new UTF8Encoding(false));
    }

    // ---------- 5. execution ----------

    [Serializable]
    public class DriveSettings
    {
        public float armStiffness = 10000f;
        public float armDamping = 100f;
        [Tooltip("0 以下なら元の forceLimit のまま")]
        public float armForceLimit = 0f;
        public float fingerStiffness = 1000f;
        public float fingerDamping = 10f;
    }

    public class PlanResult
    {
        public string cid, dir, sceneId;
        public PlannedTrajectory tr;
    }

    Vector3[] settledPos;
    Quaternion[] settledRot;
    bool[] savedKinematic;
    CollisionDetectionMode[] savedMode;
    List<(ArticulationBody body, float pos)> initialDofPositions;
    List<Transform> linkTransforms;

    List<(ArticulationBody body, float pos)> CaptureAllDofs()
    {
        var list = new List<(ArticulationBody body, float pos)>();
        foreach (var ab in robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (!ab.isRoot && ab.dofCount == 1) list.Add((ab, ab.jointPosition[0]));
        return list;
    }

    void ApplyDrives()
    {
        foreach (var (ab, pos) in initialDofPositions)
        {
            var d = ab.xDrive;
            bool arm = Array.IndexOf(kin.JointBodies, ab) >= 0;
            d.stiffness = arm ? drives.armStiffness : drives.fingerStiffness;
            d.damping = arm ? drives.armDamping : drives.fingerDamping;
            if (arm && drives.armForceLimit > 0f) d.forceLimit = drives.armForceLimit;
            d.target = pos * Mathf.Rad2Deg;
            ab.xDrive = d;
        }
    }

    // 物体と腕を「落ち着かせた直後の状態」に戻す(衝突判定モードは変えない、指の関節は触らない)
    IEnumerator ResetScene(Action<float> done)
    {
        for (int i = 0; i < bodies.Length; i++)
        {
            var b = bodies[i];
            b.isKinematic = true;
            b.position = settledPos[i];
            b.rotation = settledRot[i];
            b.transform.SetPositionAndRotation(settledPos[i], settledRot[i]);
        }
        foreach (var (ab, pos) in initialDofPositions)
        {
            if (Array.IndexOf(kin.JointBodies, ab) < 0) continue;
            ab.jointPosition = new ArticulationReducedSpace(pos);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
            var d = ab.xDrive;
            d.target = pos * Mathf.Rad2Deg;
            ab.xDrive = d;
        }
        Physics.SyncTransforms();
        for (int i = 0; i < Mathf.Max(1, resetHoldSteps); i++) yield return new WaitForFixedUpdate();

        float err = 0f;
        for (int i = 0; i < bodies.Length; i++)
        {
            var b = bodies[i];
            err = Mathf.Max(err, Vector3.Distance(b.position, settledPos[i]));
            b.isKinematic = savedKinematic[i];
            if (!b.isKinematic) { b.linearVelocity = Vector3.zero; b.angularVelocity = Vector3.zero; }
        }
        foreach (var (ab, pos) in initialDofPositions)
            err = Mathf.Max(err, Mathf.Abs(ab.jointPosition[0] - pos) * 0.1f);   // rad を m 相当に(0.1 m/rad)
        done(err);
    }

    // 物体にだけ付ける(ArticulationBody に実行中に付けると PhysX が落ちた)。
    // ロボット↔物体の接触は物体側で articulationBody として記録される
    void EnsureContactRecorders()
    {
        foreach (var b in bodies)
            if (b.GetComponent<ContactRecorder>() == null) b.gameObject.AddComponent<ContactRecorder>().selfKind = "object";
    }

    static Transform FindByName(Transform root, string name)
    {
        foreach (var t in root.GetComponentsInChildren<Transform>(true))
            if (t.name == name) return t;
        return null;
    }

    IEnumerator ExecuteCandidate(PlanResult p)
    {
        var tr = p.tr;
        float resetErr = 0f;
        yield return ResetScene(e => resetErr = e);
        Debug.Log($"[EpisodeRecorder] {p.cid} 実行開始(reset err {resetErr:E1})");

        if (linkTransforms == null)
            linkTransforms = kin.KeypointNames.Take(kin.KeypointNames.Length - 1).Select(n => FindByName(robotRoot, n)).ToList();
        string[] linkNames = kin.KeypointNames.Take(kin.KeypointNames.Length - 1).ToArray();

        var contactRows = new List<string>();
        ContactRecorder.Rows = contactRows;

        var hdr = new List<string> { "step", "t", "phase", "planned_step" };
        hdr.AddRange(kin.JointNames.Select(n => "cmd_" + n));
        hdr.AddRange(kin.JointNames.Select(n => "q_" + n));
        hdr.AddRange(kin.JointNames.Select(n => "qd_" + n));
        hdr.AddRange(new[] { "tcp_x", "tcp_y", "tcp_z", "tool_rot_x", "tool_rot_y", "tool_rot_z", "tool_rot_w" });
        foreach (var n in linkNames) hdr.AddRange(new[] { n + "_x", n + "_y", n + "_z", n + "_rx", n + "_ry", n + "_rz", n + "_rw" });
        foreach (var b in bodies) hdr.AddRange(new[] { b.name + "_x", b.name + "_y", b.name + "_z", b.name + "_rx", b.name + "_ry", b.name + "_rz", b.name + "_rw", b.name + "_vx", b.name + "_vy", b.name + "_vz" });
        var sb = new StringBuilder(string.Join(",", hdr) + "\n");

        int nPlan = tr.q.Count;
        int total = nPlan + Mathf.Max(0, postSettleSteps);
        float maxJointErr = 0f, maxTcpErr = 0f;
        double sumTcpErr2 = 0;

        for (int i = 0; i < total; i++)
        {
            int ps = Mathf.Min(i, nPlan - 1);
            var qCmd = tr.q[ps];
            for (int j = 0; j < kin.Dof; j++)
            {
                var ab = kin.JointBodies[j];
                var d = ab.xDrive;
                d.target = qCmd[j] * Mathf.Rad2Deg;
                // 計画の速度も渡す(渡さないと damping がブレーキになって追従が遅れる)
                float qd = 0f;
                if (i < nPlan - 1) qd = (tr.q[Mathf.Min(ps + 1, nPlan - 1)][j] - tr.q[Mathf.Max(ps - 1, 0)][j]) / ((Mathf.Min(ps + 1, nPlan - 1) - Mathf.Max(ps - 1, 0)) * tr.dt);
                d.targetVelocity = qd * Mathf.Rad2Deg;
                ab.xDrive = d;
            }
            ContactRecorder.CurrentStep = i;
            ContactRecorder.CurrentTime = (i + 1) * tr.dt;
            yield return new WaitForFixedUpdate();   // 物理 1 ステップ後の状態を記録する

            string phase = i < nPlan ? tr.phase[i] : "post_settle";
            var q = ReadArmJoints();
            Vector3 tcp = kin.Tool.TransformPoint(kin.TcpLocal);
            Quaternion tr0 = kin.Tool.rotation;
            if (i < nPlan)
            {
                for (int j = 0; j < kin.Dof; j++) maxJointErr = Mathf.Max(maxJointErr, Mathf.Abs(q[j] - qCmd[j]));
                float e = Vector3.Distance(tcp, tr.tcp[i]);
                maxTcpErr = Mathf.Max(maxTcpErr, e);
                sumTcpErr2 += e * e;
            }

            sb.Append(i).Append(',').Append(N((i + 1) * tr.dt)).Append(',').Append(phase).Append(',').Append(ps);
            foreach (var v in qCmd) sb.Append(',').Append(N(v));
            foreach (var v in q) sb.Append(',').Append(N(v));
            for (int j = 0; j < kin.Dof; j++) sb.Append(',').Append(N(kin.JointBodies[j].jointVelocity[0]));
            AppendPose(sb, tcp, tr0);
            foreach (var t in linkTransforms) AppendPose(sb, t.position, t.rotation);
            foreach (var b in bodies)
            {
                AppendPose(sb, b.position, b.rotation);
                var v = b.linearVelocity;
                sb.Append(',').Append(N(v.x)).Append(',').Append(N(v.y)).Append(',').Append(N(v.z));
            }
            sb.Append('\n');
        }
        ContactRecorder.Rows = null;
        ContactRecorder.CurrentStep = -1;

        string exDir = Path.Combine(p.dir, "executed");
        Directory.CreateDirectory(exDir);
        var utf8 = new UTF8Encoding(false);
        File.WriteAllText(Path.Combine(exDir, "executed_trajectory.csv"), sb.ToString(), utf8);
        File.WriteAllText(Path.Combine(exDir, "contacts.csv"),
            "step,t,event,body,body_kind,other,other_kind,num_points,px,py,pz,nx,ny,nz,impulse,rel_speed\n" +
            string.Join("\n", contactRows) + (contactRows.Count > 0 ? "\n" : ""), utf8);

        // ---- 接触のまとめ ----
        int firstRobotTarget = -1, robotTargetSteps = 0, firstTargetOther = -1, firstRobotOther = -1, firstRobotEnv = -1;
        foreach (var row in contactRows)
        {
            var f = row.Split(',');
            if (f[2] == "exit") continue;
            int st = int.Parse(f[0], CultureInfo.InvariantCulture);
            string body = f[3], bodyKind = f[4], other = f[5], otherKind = f[6];
            bool bodyIsTarget = body == targetObjectName, otherIsTarget = other == targetObjectName;
            if (bodyKind == "object" && otherKind == "robot_link")
            {
                if (bodyIsTarget) { robotTargetSteps++; if (firstRobotTarget < 0) firstRobotTarget = st; }
                else if (firstRobotOther < 0) firstRobotOther = st;
            }
            else if (bodyKind == "object" && otherKind == "object" && (bodyIsTarget || otherIsTarget))
            {
                if (firstTargetOther < 0) firstTargetOther = st;
            }
            else if (bodyKind == "robot_link" && otherKind == "environment")
            {
                if (firstRobotEnv < 0) firstRobotEnv = st;
            }
        }
        float rmsTcp = nPlan > 0 ? Mathf.Sqrt((float)(sumTcpErr2 / nPlan)) : 0f;

        string meta = "{\n" +
            "  \"schema\": \"executed_trajectory_v1\",\n" +
            $"  \"scene_id\": {Q(p.sceneId)},\n" +
            $"  \"candidate_id\": {Q(p.cid)},\n" +
            "  \"command_source\": \"../planned_trajectory.json (joint position targets, one per physics step)\",\n" +
            "  \"timing\": \"row step i = state measured after physics step i with command planned_step; t = (i + 1) * dt\",\n" +
            $"  \"dt\": {N(tr.dt)},\n" +
            $"  \"num_planned_steps\": {nPlan},\n" +
            $"  \"post_settle_steps\": {Mathf.Max(0, postSettleSteps)},\n" +
            $"  \"num_steps\": {total},\n" +
            $"  \"drives\": {JsonUtility.ToJson(drives)},\n" +
            $"  \"reset\": {{\"hold_steps\": {resetHoldSteps}, \"max_error\": {N(resetErr)}}},\n" +
            $"  \"tracking\": {{\"max_joint_err_rad\": {N(maxJointErr)}, \"max_tcp_err_m\": {N(maxTcpErr)}, \"rms_tcp_err_m\": {N(rmsTcp)}}},\n" +
            "  \"contact_summary\": {" +
            $"\"num_rows\": {contactRows.Count}, \"first_robot_target_step\": {firstRobotTarget}, \"robot_target_steps\": {robotTargetSteps}, " +
            $"\"first_target_object_contact_step\": {firstTargetOther}, \"first_robot_other_object_step\": {firstRobotOther}, " +
            $"\"first_robot_environment_step\": {firstRobotEnv}}},\n" +
            "  \"files\": {\"trajectory\": \"executed_trajectory.csv\", \"contacts\": \"contacts.csv\"},\n" +
            "  \"columns\": \"cmd_* = commanded joint target (rad), q_* / qd_* = measured joint position / velocity, tcp/tool_rot = measured TCP pose, " +
            "<link>_* = measured link origin pose, <object>_* = measured object pose and linear velocity; rotations are quaternions xyzw; world = Unity (left-handed, Y-up, m)\"\n" +
            "}\n";
        File.WriteAllText(Path.Combine(exDir, "executed_meta.json"), meta, utf8);

        Debug.Log($"[EpisodeRecorder] {p.cid} 実行完了: {total} steps, reset err {resetErr:E1}, 追従誤差 最大 {maxJointErr * Mathf.Rad2Deg:F2}° / TCP 最大 {maxTcpErr * 1000f:F1} mm(RMS {rmsTcp * 1000f:F1} mm), " +
                  $"接触: robot-target 初回 step {firstRobotTarget}({robotTargetSteps} steps), target-物体 初回 {firstTargetOther}, robot-他物体 初回 {firstRobotOther}, robot-環境 初回 {firstRobotEnv}");

        yield return SaveFinal(p, firstRobotTarget, robotTargetSteps, firstTargetOther, firstRobotOther);
    }

    // ---------- 6. final ----------

    IEnumerator SaveFinal(PlanResult p, int firstRobotTarget, int robotTargetSteps, int firstTargetOther, int firstRobotOther)
    {
        string fDir = Path.Combine(p.dir, "final");
        Directory.CreateDirectory(fDir);

        // 撮影中に動かないよう固定してから、最終姿勢を記録
        for (int i = 0; i < bodies.Length; i++) bodies[i].isKinematic = true;
        var finalPos = bodies.Select(b => b.position).ToArray();
        var finalRot = bodies.Select(b => b.rotation).ToArray();

        int stepWith = captureIndex;
        yield return CaptureOne();
        SetRobotVisible(false);
        for (int i = 0; i < hideSettleFrames; i++) yield return null;
        int stepFree = captureIndex;
        yield return CaptureOne();
        SetRobotVisible(true);

        bool okWith = false, okFree = false;
        if (runSoloDir != null)
        {
            yield return CopyStep(runSoloDir, stepWith, Path.Combine(fDir, "with_robot"), r => okWith = r);
            yield return CopyStep(runSoloDir, stepFree, Path.Combine(fDir, "robot_free"), r => okFree = r);
        }
        int maskPixels = -1;
        string sem = Path.Combine(fDir, "robot_free", "semantic.png");
        if (okFree && File.Exists(sem)) maskPixels = WriteTargetMask(sem, Path.Combine(fDir, "target_mask.png"));

        // ---- ラベル ----
        int ti = Array.FindIndex(bodies, b => b.name == targetObjectName);
        Vector3 t1 = finalPos[ti];
        float goalDist = new Vector2(t1.x - goalCenter.x, t1.z - goalCenter.z).magnitude;
        bool fell = t1.y < settledPos[ti].y - fallDropThreshold;
        bool goal = goalDist <= goalRadius && !fell;
        bool targetContact = firstRobotTarget >= 0;
        bool secondary = firstTargetOther >= 0 || firstRobotOther >= 0;

        var objs = new List<string>();
        for (int i = 0; i < bodies.Length; i++)
        {
            objs.Add("    {" +
                $"\"name\": {Q(bodies[i].name)}, " +
                $"\"initial_position_world\": {Vec(settledPos[i])}, \"initial_rotation_world_xyzw\": {Quat(settledRot[i])}, " +
                $"\"final_position_world\": {Vec(finalPos[i])}, \"final_rotation_world_xyzw\": {Quat(finalRot[i])}, " +
                $"\"displacement_m\": {N(Vector3.Distance(finalPos[i], settledPos[i]))}, " +
                $"\"rotation_deg\": {N(Quaternion.Angle(settledRot[i], finalRot[i]))}}}");
        }

        string json = "{\n" +
            "  \"schema\": \"final_state_v1\",\n" +
            $"  \"scene_id\": {Q(p.sceneId)},\n" +
            $"  \"candidate_id\": {Q(p.cid)},\n" +
            "  \"measured_after\": \"end of post_settle steps (objects frozen for the final capture)\",\n" +
            "  \"objects\": [\n" + string.Join(",\n", objs) + "\n  ],\n" +
            $"  \"goal\": {{\"center_world\": {Vec(goalCenter)}, \"radius_m\": {N(goalRadius)}, \"definition\": \"target final center within radius of goal center in XZ, and not fallen\"}},\n" +
            "  \"labels\": {" +
            $"\"goal_reached\": {B(goal)}, \"goal_distance_m\": {N(goalDist)}, \"target_fell\": {B(fell)}, " +
            $"\"target_contact\": {B(targetContact)}, \"first_robot_target_step\": {firstRobotTarget}, \"robot_target_contact_rows\": {robotTargetSteps}, " +
            $"\"secondary_collision\": {B(secondary)}, \"first_target_secondary_step\": {firstTargetOther}, \"first_robot_secondary_step\": {firstRobotOther}}},\n" +
            $"  \"label_params\": {{\"fall_drop_threshold_m\": {N(fallDropThreshold)}}},\n" +
            $"  \"target_mask\": {{\"file\": \"target_mask.png\", \"pixels\": {maskPixels}, \"semantic_color_rgb\": [{targetMaskColor.r}, {targetMaskColor.g}, {targetMaskColor.b}], \"encoding\": \"8-bit gray, 255 = target\"}},\n" +
            $"  \"capture\": {{\"step_with_robot\": {stepWith}, \"step_robot_free\": {stepFree}, \"copied_with_robot\": {B(okWith)}, \"copied_robot_free\": {B(okFree)}}},\n" +
            "  \"files\": {\"robot_free\": {\"rgb\": \"robot_free/rgb.png\", \"depth\": \"robot_free/depth.exr\", \"semantic\": \"robot_free/semantic.png\"}, " +
            "\"with_robot\": {\"rgb\": \"with_robot/rgb.png\", \"depth\": \"with_robot/depth.exr\", \"semantic\": \"with_robot/semantic.png\"}, \"target_mask\": \"target_mask.png\"}\n" +
            "}\n";
        File.WriteAllText(Path.Combine(fDir, "final_state.json"), json, new UTF8Encoding(false));

        for (int i = 0; i < bodies.Length; i++) bodies[i].isKinematic = savedKinematic[i];

        Debug.Log($"[EpisodeRecorder] {p.cid} 最終結果: goal={goal}(距離 {goalDist * 1000f:F1} mm), fell={fell}, target_contact={targetContact}, secondary={secondary}, " +
                  $"target 移動 {Vector3.Distance(finalPos[ti], settledPos[ti]) * 1000f:F1} mm, mask {maskPixels} px, 画像 with={okWith} free={okFree}");
    }

    int WriteTargetMask(string semPath, string outPath)
    {
        var tex = new Texture2D(2, 2);
        if (!tex.LoadImage(File.ReadAllBytes(semPath))) return -1;
        var px = tex.GetPixels32();
        var m = new byte[px.Length];
        int n = 0;
        for (int i = 0; i < px.Length; i++)
            if (px[i].r == targetMaskColor.r && px[i].g == targetMaskColor.g && px[i].b == targetMaskColor.b) { m[i] = 255; n++; }
        // LoadImage も EncodeArrayToPNG も下の行から並ぶので、そのまま渡せば上下は元画像と同じ
        File.WriteAllBytes(outPath, ImageConversion.EncodeArrayToPNG(m, UnityEngine.Experimental.Rendering.GraphicsFormat.R8_UNorm, (uint)tex.width, (uint)tex.height));
        Destroy(tex);
        return n;
    }

    static string B(bool b) => b ? "true" : "false";

    static void AppendPose(StringBuilder sb, Vector3 p, Quaternion r)
    {
        sb.Append(',').Append(N(p.x)).Append(',').Append(N(p.y)).Append(',').Append(N(p.z))
          .Append(',').Append(N(r.x)).Append(',').Append(N(r.y)).Append(',').Append(N(r.z)).Append(',').Append(N(r.w));
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

    // ---------- 2. JSON ----------

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

    static string N(double d) => d.ToString("G9", CultureInfo.InvariantCulture);
    static string Q(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    static string KV(string k, string v) => $"  \"{k}\": {v}";
    static string Vec(Vector3 v) => $"[{N(v.x)}, {N(v.y)}, {N(v.z)}]";
    static string Quat(Quaternion q) => $"[{N(q.x)}, {N(q.y)}, {N(q.z)}, {N(q.w)}]";
}
