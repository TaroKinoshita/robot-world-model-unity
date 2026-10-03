using UnityEngine;

/// <summary>
/// Play 開始時にロボットの腕を指定の初期姿勢へ瞬時にセットする(ドライブの目標も同じ角度にする)。
/// UR5e (ur_e_description) は全関節 0 だと腕が水平に伸びるため、[0,-90,0,-90,0,0] deg で
/// UR3 の全関節 0 と同じ「真上に立った姿勢」にする。EpisodeRecorder はこの姿勢を初期姿勢として記録・使用する。
/// </summary>
[DefaultExecutionOrder(-100)]
public class RobotInitialPose : MonoBehaviour
{
    public string[] jointLinkNames = { "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link" };
    public float[] jointDegrees = { 0f, -90f, 0f, -90f, 0f, 0f };

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
        Debug.Log("[RobotInitialPose] set: [" + string.Join(", ", System.Array.ConvertAll(jointDegrees, x => x.ToString("F1"))) + "] deg");
    }

    static Transform FindDeep(Transform t, string name)
    {
        if (t.name == name) return t;
        foreach (Transform c in t) { var r = FindDeep(c, name); if (r != null) return r; }
        return null;
    }
}
