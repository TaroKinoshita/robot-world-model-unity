using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

/// <summary>
/// ロボットあり / ロボットなし(robot-free) をペアで撮影する。
/// 前提: PerceptionCamera の Capture Trigger Mode を Inspector で Manual にしておくこと。
/// SOLO の step 番号 = このスクリプトの capture_index(0始まり)になる想定。
/// </summary>
public class RobotFreeCapture : MonoBehaviour
{
    [Header("References")]
    public PerceptionCamera perceptionCamera;
    public Transform robotRoot;              // ur3_with_gripper を入れる

    [Header("Scene objects")]
    [Tooltip("撮影中は物体のRigidbodyを固定する(レンダリング検証専用)。物理で押す実行ではオフにする")]
    public bool freezeObjectsDuringCapture = true;
    public Transform objectsRoot;            // Objects を入れる。空なら名前 "Objects" で探す
    Rigidbody[] frozenBodies;
    bool[] originalKinematic;
    CollisionDetectionMode[] originalCollisionMode;

    [Header("Timing (frames)")]
    public int framesBeforeStart = 30;       // ドライブが落ち着くまで待つ
    public int hideSettleFrames = 1;         // 非表示にしてから撮影までの待ち
    public int framesBetweenPairs = 30;      // ペア間の間隔

    [Header("Run")]
    public int numPairs = 5;                 // poses が空のときだけ使う

    [Header("Poses (検証B用・度)")]
    public int poseSettleFrames = 60;        // 姿勢変更後に落ち着くまで待つ
    public string[] armJointNames = {
        "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link" };
    // 検証A用: shoulderを45°刻みで一周 × 2種類の前傾 = 16姿勢
    public List<ArmPose> poses = BuildSweep();

    static List<ArmPose> BuildSweep()
    {
        var list = new List<ArmPose>();
        float[][] leans = {
            new float[] { -60f, 100f, -40f },   // 軽く前傾
            new float[] { -120f, 120f, 0f },    // 大きく前傾
        };
        foreach (var lean in leans)
            for (int pan = 0; pan < 360; pan += 45)
                list.Add(new ArmPose(pan, lean[0], lean[1], lean[2], 0f, 0f));
        return list;
    }

    [System.Serializable]
    public class ArmPose
    {
        public float[] deg = new float[6];
        public ArmPose() { }
        public ArmPose(params float[] d) { deg = d; }
    }

    Renderer[] robotRenderers;
    bool[] originalForceOff;
    ArticulationBody[] joints;
    StreamWriter log;
    int captureIndex = 0;

    void Start()
    {
        if (perceptionCamera == null || robotRoot == null)
        {
            Debug.LogError("[RobotFreeCapture] perceptionCamera と robotRoot を設定して");
            enabled = false;
            return;
        }

        // ロボットの全Renderer(非アクティブ含む)を取得し、元の状態を保存
        robotRenderers = robotRoot.GetComponentsInChildren<Renderer>(true);
        originalForceOff = new bool[robotRenderers.Length];
        for (int i = 0; i < robotRenderers.Length; i++)
            originalForceOff[i] = robotRenderers[i].forceRenderingOff;

        // 物体を固定(撮影中に腕が当たっても動かないように)
        if (freezeObjectsDuringCapture) FreezeObjects();

        // 自由度のある関節だけ記録対象にする
        var jointList = new List<ArticulationBody>();
        foreach (var ab in robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (!ab.isRoot && ab.dofCount > 0) jointList.Add(ab);
        joints = jointList.ToArray();

        OpenLog();
        Debug.Log($"[RobotFreeCapture] renderers={robotRenderers.Length}, joints={joints.Length}");
        StartCoroutine(Run());
    }

    IEnumerator Run()
    {
        yield return WaitFrames(framesBeforeStart);

        int pairs = (poses != null && poses.Count > 0) ? poses.Count : numPairs;
        for (int p = 0; p < pairs; p++)
        {
            // 0) 姿勢を設定(poses があるときだけ)
            if (poses != null && poses.Count > 0)
            {
                ApplyPose(poses[p]);
                yield return WaitFrames(poseSettleFrames);
            }

            // 1) ロボットあり
            yield return CaptureOne(p, true);

            // 2) ロボットを消す → 待つ → robot-free撮影
            SetRobotVisible(false);
            yield return WaitFrames(hideSettleFrames);
            yield return CaptureOne(p, false);

            // 3) 表示に戻す
            SetRobotVisible(true);
            yield return WaitFrames(framesBetweenPairs);
        }

        log.Flush();
        log.Close();
        log = null;
        Debug.Log($"[RobotFreeCapture] 完了: {captureIndex} captures");
    }

    IEnumerator CaptureOne(int pairIndex, bool robotVisible)
    {
        perceptionCamera.RequestCapture();   // このフレームの描画が撮影される
        WriteLogRow(pairIndex, robotVisible);
        captureIndex++;
        yield return null;                   // 撮影フレームを描画させる
    }

    void SetRobotVisible(bool visible)
    {
        for (int i = 0; i < robotRenderers.Length; i++)
        {
            if (robotRenderers[i] == null) continue; // Play終了時などに破棄済みのものはスキップ
            robotRenderers[i].forceRenderingOff = visible ? originalForceOff[i] : true;
        }
    }

    void ApplyPose(ArmPose pose)
    {
        for (int k = 0; k < armJointNames.Length && k < pose.deg.Length; k++)
        {
            ArticulationBody ab = null;
            foreach (var j in joints) if (j.name == armJointNames[k]) { ab = j; break; }
            if (ab == null) { Debug.LogWarning($"[RobotFreeCapture] joint not found: {armJointNames[k]}"); continue; }

            // ドライブの目標(度)を設定
            var drive = ab.xDrive;
            drive.target = pose.deg[k];
            ab.xDrive = drive;

            // 即座にその角度へワープ(ラジアン)して、速度はゼロに
            ab.jointPosition = new ArticulationReducedSpace(pose.deg[k] * Mathf.Deg2Rad);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
        }
    }

    IEnumerator WaitFrames(int n)
    {
        for (int i = 0; i < n; i++) yield return null;
    }

    // ---------- logging ----------

    void OpenLog()
    {
        string dir = Path.Combine(Application.dataPath, "..", "CaptureLogs");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"robot_free_log_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv");
        log = new StreamWriter(path, false, Encoding.UTF8);

        var header = new StringBuilder("capture_index,pair_index,robot_visible,unity_frame");
        foreach (var j in joints) header.Append(",").Append(j.name);
        log.WriteLine(header.ToString());
        Debug.Log($"[RobotFreeCapture] log: {Path.GetFullPath(path)}");
    }

    void WriteLogRow(int pairIndex, bool robotVisible)
    {
        var row = new StringBuilder();
        row.Append(captureIndex).Append(",")
           .Append(pairIndex).Append(",")
           .Append(robotVisible ? 1 : 0).Append(",")
           .Append(Time.frameCount);
        foreach (var j in joints)
            row.Append(",").Append(j.jointPosition[0].ToString("F6")); // rad(回転) / m(直動)
        log.WriteLine(row.ToString());
    }

    // ---------- scene objects ----------

    void FreezeObjects()
    {
        if (objectsRoot == null)
        {
            var go = GameObject.Find("Objects");
            if (go != null) objectsRoot = go.transform;
        }
        if (objectsRoot == null)
        {
            Debug.LogWarning("[RobotFreeCapture] objectsRoot が見つからない。物体は固定しない");
            return;
        }

        frozenBodies = objectsRoot.GetComponentsInChildren<Rigidbody>(true);
        originalKinematic = new bool[frozenBodies.Length];
        originalCollisionMode = new CollisionDetectionMode[frozenBodies.Length];
        for (int i = 0; i < frozenBodies.Length; i++)
        {
            var rb = frozenBodies[i];
            originalKinematic[i] = rb.isKinematic;
            originalCollisionMode[i] = rb.collisionDetectionMode;
            // kinematic は ContinuousSpeculative しか使えないので先に切り替える
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
            rb.isKinematic = true;
        }
        Debug.Log($"[RobotFreeCapture] 物体を固定: {frozenBodies.Length} bodies");
    }

    void RestoreObjects()
    {
        if (frozenBodies == null) return;
        for (int i = 0; i < frozenBodies.Length; i++)
        {
            var rb = frozenBodies[i];
            if (rb == null) continue;
            rb.isKinematic = originalKinematic[i];
            rb.collisionDetectionMode = originalCollisionMode[i];
        }
        frozenBodies = null;
    }

    void OnDestroy()
    {
        if (robotRenderers != null) SetRobotVisible(true);
        RestoreObjects();
        if (log != null) { log.Flush(); log.Close(); }
    }
}