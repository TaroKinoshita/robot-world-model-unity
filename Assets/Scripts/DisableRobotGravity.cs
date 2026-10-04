using UnityEngine;

/// <summary>
/// ロボットの重力を切り、Play 開始時のドライブ値を入れる。
/// P5: 値は EpisodeRecorder.drives(episode の実行中に使う値)の腕と同じにした(腕 20000 / 200 / 5000)。
/// 指は EpisodeRecorder.ApplyDrives が実行前に 1000 / 10 へ上書きする。
/// </summary>
public class DisableRobotGravity : MonoBehaviour
{
    public float stiffness = 20000f;
    public float damping = 200f;
    public float forceLimit = 5000f;

    void Awake()
    {
        foreach (var ab in GetComponentsInChildren<ArticulationBody>(true))
        {
            ab.useGravity = false;

            if (ab.isRoot) continue; // ルートは関節じゃないのでスキップ

            var drive = ab.xDrive;
            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = forceLimit;
            ab.xDrive = drive;
        }
    }
}