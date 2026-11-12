using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 2 つ目のタスク専用。棚の板・かごに付けて、ロボットのリンク（ArticulationBody）が触れたことを PhysX の通知で記録する。
/// 触れていなくても近づくと通知が来るので（contactOffset）、隙間 maxSeparation 以下の接触点があるときだけ数える。
/// </summary>
public class ShelfContactProbe : MonoBehaviour
{
    public static readonly List<(string link, string board, float sep, Vector3 point)> Hits = new List<(string, string, float, Vector3)>();
    public static bool Enabled;
    public static float maxSeparation = 0.001f;

    void OnCollisionEnter(Collision c) => Rec(c);
    void OnCollisionStay(Collision c) => Rec(c);

    void Rec(Collision c)
    {
        if (!Enabled || c.articulationBody == null) return;
        float min = float.PositiveInfinity; Vector3 p = Vector3.zero;
        for (int i = 0; i < c.contactCount; i++) { var cp = c.GetContact(i); if (cp.separation < min) { min = cp.separation; p = cp.point; } }
        if (min <= maxSeparation) Hits.Add((c.articulationBody.name, name, min, p));
    }
}
