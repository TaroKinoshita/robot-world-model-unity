using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

/// <summary>
/// 衝突コールバックを 1 行ずつ記録する(EpisodeRecorder が実行中だけ Rows をセットする)。
/// 物体に付ける: ロボット・他の物体・環境(机など)との接触を記録。物体同士は名前順で片側だけ記録する。
/// ロボットのリンクに付ける(onlyEnvironment = true): 環境との接触だけ記録(物体との接触は物体側で記録済み)。
/// 行: step,t,event,body,body_kind,other,other_kind,num_points,px,py,pz,nx,ny,nz,impulse,rel_speed
/// </summary>
public class ContactRecorder : MonoBehaviour
{
    public static List<string> Rows;       // null なら記録しない
    public static int CurrentStep = -1;
    public static float CurrentTime;

    public string selfKind = "object";
    public bool onlyEnvironment = false;

    void OnCollisionEnter(Collision c) => Record("enter", c);
    void OnCollisionStay(Collision c) => Record("stay", c);
    void OnCollisionExit(Collision c) => Record("exit", c);

    void Record(string ev, Collision c)
    {
        if (Rows == null || CurrentStep < 0) return;

        string otherKind, otherName;
        if (c.articulationBody != null) { otherKind = "robot_link"; otherName = c.articulationBody.name; }
        else if (c.rigidbody != null) { otherKind = "object"; otherName = c.rigidbody.name; }
        else { otherKind = "environment"; otherName = c.collider.gameObject.name; }

        if (onlyEnvironment && otherKind != "environment") return;
        if (selfKind == "robot_link" && otherKind == "robot_link") return;
        if (selfKind == "object" && otherKind == "object" && string.CompareOrdinal(name, otherName) > 0) return;

        int n = c.contactCount;
        Vector3 p = Vector3.zero, nrm = Vector3.zero;
        for (int i = 0; i < n; i++) { var cp = c.GetContact(i); p += cp.point; nrm += cp.normal; }
        if (n > 0) { p /= n; nrm = nrm.normalized; }

        Rows.Add(string.Join(",", new[]
        {
            CurrentStep.ToString(CultureInfo.InvariantCulture), N(CurrentTime), ev, name, selfKind, otherName, otherKind,
            n.ToString(CultureInfo.InvariantCulture), N(p.x), N(p.y), N(p.z), N(nrm.x), N(nrm.y), N(nrm.z),
            N(c.impulse.magnitude), N(c.relativeVelocity.magnitude)
        }));
    }

    static string N(float v) => v.ToString("G7", CultureInfo.InvariantCulture);
}
