using UnityEngine;

public class DisableRobotGravity : MonoBehaviour
{
    public float stiffness = 10000f;
    public float damping = 100f;
    public float forceLimit = 1000f;

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