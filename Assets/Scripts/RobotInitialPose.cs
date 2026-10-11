using UnityEngine;

/// <summary>
/// Play 開始時にロボットの腕を指定の初期姿勢へ瞬時にセットする(ドライブの目標も同じ角度にする)。
/// UR5e は全関節 0 だと腕が水平に約 1 m 伸びるため、初期姿勢をここで決める。
/// 今の値はシーンに保存してある [-63.5, -151.8, 122.3, -60.5, -90, -63.5] deg(手先 (0, 1.05, 0.15)、wrist-down の枝。
/// 腕全体と手先が画面に入る低い待機姿勢、TODO6 で決定)。下の既定値 [0,-90,0,-90,0,0](真上に立つ形)は最初の版のまま。
/// 指は gripperCloseRad(シーンでは 0.42 rad = パッドの間隔約 6 cm、P4-B)。
/// EpisodeRecorder はこの姿勢を初期姿勢として記録・使用する。
/// </summary>
[DefaultExecutionOrder(-100)]
public class RobotInitialPose : MonoBehaviour
{
    public string[] jointLinkNames = { "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link" };
    public float[] jointDegrees = { 0f, -90f, 0f, -90f, 0f, 0f };

    [Header("Gripper (P4-B)")]
    [Tooltip("finger_joint の角度(rad)。0 = 全開、0.7 = 全閉。他の指の関節は URDF の mimic の倍率で決める")]
    public float gripperCloseRad = 0f;
    public string[] gripperLinkNames = { "left_outer_knuckle", "left_inner_knuckle", "left_inner_finger",
                                         "right_outer_knuckle", "right_inner_knuckle", "right_inner_finger" };
    [Tooltip("URDF の mimic の倍率(finger_joint に対して)")]
    public float[] gripperMimic = { 1f, -1f, 1f, -1f, -1f, 1f };

    void Start()
    {
        if (jointDegrees.Length != jointLinkNames.Length)
        {
            Debug.LogWarning("[RobotInitialPose] jointDegrees と jointLinkNames の数が違う");
            return;
        }
        for (int i = 0; i < jointLinkNames.Length; i++)
        {
            var t = FindDeep(transform, jointLinkNames[i]);
            var ab = t != null ? t.GetComponent<ArticulationBody>() : null;
            if (ab == null) { Debug.LogWarning($"[RobotInitialPose] {jointLinkNames[i]} が見つからない"); continue; }
            var d = ab.xDrive; d.target = jointDegrees[i]; ab.xDrive = d;
            ab.jointPosition = new ArticulationReducedSpace(jointDegrees[i] * Mathf.Deg2Rad);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
        }
        if (gripperCloseRad != 0f && gripperLinkNames.Length == gripperMimic.Length)
        {
            for (int i = 0; i < gripperLinkNames.Length; i++)
            {
                var t = FindDeep(transform, gripperLinkNames[i]);
                var ab = t != null ? t.GetComponent<ArticulationBody>() : null;
                if (ab == null) { Debug.LogWarning($"[RobotInitialPose] {gripperLinkNames[i]} が見つからない"); continue; }
                float q = gripperCloseRad * gripperMimic[i];
                var d = ab.xDrive; d.target = q * Mathf.Rad2Deg; ab.xDrive = d;
                ab.jointPosition = new ArticulationReducedSpace(q);
                ab.jointVelocity = new ArticulationReducedSpace(0f);
            }
            Debug.Log($"[RobotInitialPose] gripper: finger_joint = {gripperCloseRad:F3} rad");
        }
        Debug.Log("[RobotInitialPose] set: [" + string.Join(", ", System.Array.ConvertAll(jointDegrees, x => x.ToString("F1"))) + "] deg");
    }

    static Transform FindDeep(Transform t, string name)
    {
        if (t.name == name) return t;
        foreach (Transform c in t) { var r = FindDeep(c, name); if (r != null) return r; }
        return null;
    }
}
