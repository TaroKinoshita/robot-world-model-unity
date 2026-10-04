using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

/// <summary>1 本の候補軌道の指定(押す方向と距離)</summary>
[Serializable]
public class CandidateSpec
{
    public string note = "";
    [Tooltip("押す方向。XZ 平面で +Z から +X 向きに測った角度(度)")]
    public float pushAngleDeg = 0f;
    [Tooltip("ターゲットを押し進める距離(m)")]
    public float pushLength = 0.10f;

    [Header("Matched pair (P3)")]
    [Tooltip("開始姿勢の IK の枝を選ぶ初期値(度、関節 6 個)。空なら初期姿勢そのまま。" +
             "初期姿勢と同じ手先位置・向きのまま、別の枝(手首の反転など)から始める")]
    public float[] startBranchSeedDeg = new float[0];
    [Tooltip("on: 開始 → 押す位置の上 を手先の直線(関節空間ではなく)で動かす。matched pair は両方 on にして手先経路を全区間そろえる")]
    public bool cartesianTransfer = false;
    [Tooltip("同じ手先経路を共有する候補に同じ ID を付ける(空 = matched pair ではない)")]
    public string matchedPairId = "";
    [Tooltip("例: wrist_down / wrist_up")]
    public string branchLabel = "";
}

/// <summary>軌道の作り方の共通設定</summary>
[Serializable]
public class PlannerSettings
{
    public float approachHeight = 0.12f;    // 押す位置の上、どれだけ高いところから降りるか(m)
    public float preContactGap = 0.05f;     // 接触の手前でいったん止まる距離(m)
    public float contactPadding = 0.015f;   // 指パッドの厚みの分の余裕(m)
    public float pushHeightOffset = -0.01f; // TCP の高さ = ターゲット中心 + これ(m)
    public float transferDuration = 3.0f;   // 初期姿勢 → 押す位置の上 までの時間(s、関節空間で補間)
    public float moveSpeed = 0.10f;         // 降りる・持ち上げるときの平均速度(m/s)
    public float pushSpeed = 0.05f;         // 押すときの平均速度(m/s)
    public float holdDuration = 0.5f;       // 最後に止まっている時間(s)
}

/// <summary>計画した軌道(物理 1 ステップごと)</summary>
public class PlannedTrajectory
{
    public bool ok;
    public string error;
    public float dt;
    public readonly List<float> t = new List<float>();
    public readonly List<string> phase = new List<string>();
    public readonly List<float[]> q = new List<float[]>();
    public readonly List<Vector3> tcp = new List<Vector3>();
    public readonly List<Quaternion> toolRot = new List<Quaternion>();
    public Vector3 pushDir;
    public string[] waypointNames = { "above", "pre_contact", "push_end", "lift" };
    public Vector3[] waypoints = new Vector3[4];
    public float maxIkPosErr, maxIkAngErr, maxJointStep;
    public float minTcpHeightTransfer = float.MaxValue;
}

public static class TrajectoryPlanner
{
    static float MinJerk(float s) => s * s * s * (10f + s * (-15f + 6f * s));

    public static PlannedTrajectory Plan(ArmKinematics kin, float[] qInit, Vector3 targetCenter, Vector3 targetHalfExtents,
                                         CandidateSpec spec, PlannerSettings s, float dt)
    {
        var tr = new PlannedTrajectory { dt = dt };

        // ---- 押す方向と通過点 ----
        float phi = spec.pushAngleDeg * Mathf.Deg2Rad;
        Vector3 u = new Vector3(Mathf.Sin(phi), 0f, Mathf.Cos(phi));
        tr.pushDir = u;
        float halfU = Mathf.Abs(u.x) * targetHalfExtents.x + Mathf.Abs(u.z) * targetHalfExtents.z;
        // 指の開閉方向 = 押す方向 → 前側の指パッドで押す。TCP は指の中点なので、その分だけ手前に置く
        float contact = halfU + kin.FingerHalfSpan + s.contactPadding;

        Vector3 pre = targetCenter - u * (contact + s.preContactGap);
        pre.y = targetCenter.y + s.pushHeightOffset;
        Vector3 above = pre + Vector3.up * s.approachHeight;
        Vector3 end = pre + u * (s.preContactGap + spec.pushLength);
        Vector3 lift = end + Vector3.up * s.approachHeight;
        tr.waypoints = new[] { above, pre, end, lift };

        Quaternion rotDes = kin.ToolRotationFor(Vector3.down, u);

        // ---- 0) 開始姿勢(matched pair では、同じ手先の位置・向きのまま別の IK の枝から始める) ----
        float[] qStart = qInit;
        if (spec.startBranchSeedDeg != null && spec.startBranchSeedDeg.Length == kin.Dof)
        {
            kin.TcpPose(qInit, out var p0, out var r0);
            var seed = spec.startBranchSeedDeg.Select(d => d * Mathf.Deg2Rad).ToArray();
            if (!kin.SolveIK(p0, r0, seed, out qStart, out float peS, out float aeS))
            {
                tr.error = $"開始姿勢の枝の IK が解けない(pos err {peS:F4} m, ang err {aeS:F4} rad)";
                return tr;
            }
            tr.maxIkPosErr = Mathf.Max(tr.maxIkPosErr, peS);
            tr.maxIkAngErr = Mathf.Max(tr.maxIkAngErr, aeS);
        }

        Add(tr, kin, qStart, "start");
        int nT = Mathf.Max(1, Mathf.RoundToInt(s.transferDuration / dt));

        if (spec.cartesianTransfer)
        {
            // ---- 1+2) 開始 → above を手先の直線で(向きは slerp)。前の step の解から続けて解くので枝は変わらない ----
            kin.TcpPose(qStart, out var p0, out var r0);
            if (!Line(tr, kin, p0, above, r0, rotDes, nT, "transfer")) return tr;
            for (int i = 1; i < tr.tcp.Count; i++) tr.minTcpHeightTransfer = Mathf.Min(tr.minTcpHeightTransfer, tr.tcp[i].y);
        }
        else
        {
            // ---- 1) 押す位置の上の姿勢(複数の初期値から IK) ----
            if (!SolveMultiSeed(kin, above, rotDes, qStart, out var qAbove, out float pe0, out float ae0))
            {
                tr.error = $"'above' の IK が解けない(pos err {pe0:F4} m, ang err {ae0:F4} rad)";
                return tr;
            }

            // ---- 2) 初期姿勢 → above(関節空間で min-jerk 補間) ----
            for (int k = 1; k <= nT; k++)
            {
                float a = MinJerk((float)k / nT);
                var qk = new float[kin.Dof];
                for (int j = 0; j < kin.Dof; j++) qk[j] = Mathf.Lerp(qStart[j], qAbove[j], a);
                Add(tr, kin, qk, "transfer");
                tr.minTcpHeightTransfer = Mathf.Min(tr.minTcpHeightTransfer, tr.tcp[tr.tcp.Count - 1].y);
            }
        }

        // ---- 3) 降りる → 押す → 持ち上げる(手先を直線で動かす) ----
        if (!Line(tr, kin, above, pre, rotDes, s.moveSpeed, "descend")) return tr;
        if (!Line(tr, kin, pre, end, rotDes, s.pushSpeed, "push")) return tr;
        if (!Line(tr, kin, end, lift, rotDes, s.moveSpeed, "lift")) return tr;

        // ---- 4) 止まる ----
        int nH = Mathf.Max(0, Mathf.RoundToInt(s.holdDuration / dt));
        var qLast = tr.q[tr.q.Count - 1];
        for (int k = 0; k < nH; k++) Add(tr, kin, (float[])qLast.Clone(), "hold");

        tr.ok = true;
        return tr;
    }

    static bool Line(PlannedTrajectory tr, ArmKinematics kin, Vector3 from, Vector3 to, Quaternion rot,
                     float speed, string phase)
    {
        float dist = Vector3.Distance(from, to);
        int n = Mathf.Max(1, Mathf.CeilToInt(dist / speed / tr.dt));
        return Line(tr, kin, from, to, rot, rot, n, phase);
    }

    // 手先を from → to へ直線(min-jerk)、向きは rotFrom → rotTo を slerp。n step
    static bool Line(PlannedTrajectory tr, ArmKinematics kin, Vector3 from, Vector3 to, Quaternion rotFrom, Quaternion rotTo,
                     int n, string phase)
    {
        var qPrev = tr.q[tr.q.Count - 1];
        for (int k = 1; k <= n; k++)
        {
            float a = MinJerk((float)k / n);
            Vector3 p = Vector3.Lerp(from, to, a);
            Quaternion rot = Quaternion.Slerp(rotFrom, rotTo, a);
            if (!kin.SolveIK(p, rot, qPrev, out var qk, out float pe, out float ae))
            {
                tr.error = $"'{phase}' の {k}/{n} 点目で IK が解けない(pos err {pe:F4} m, ang err {ae:F4} rad)";
                return false;
            }
            tr.maxIkPosErr = Mathf.Max(tr.maxIkPosErr, pe);
            tr.maxIkAngErr = Mathf.Max(tr.maxIkAngErr, ae);
            Add(tr, kin, qk, phase);
            qPrev = qk;
        }
        return true;
    }

    static void Add(PlannedTrajectory tr, ArmKinematics kin, float[] q, string phase)
    {
        if (tr.q.Count > 0)
        {
            var prev = tr.q[tr.q.Count - 1];
            for (int j = 0; j < q.Length; j++) tr.maxJointStep = Mathf.Max(tr.maxJointStep, Mathf.Abs(q[j] - prev[j]));
        }
        kin.TcpPose(q, out var tcp, out var rot);
        tr.t.Add(tr.q.Count * tr.dt);
        tr.phase.Add(phase);
        tr.q.Add(q);
        tr.tcp.Add(tcp);
        tr.toolRot.Add(rot);
    }

    /// <summary>いくつかの初期姿勢から IK を解き、初期姿勢からの関節の動きが一番小さい解を選ぶ</summary>
    static bool SolveMultiSeed(ArmKinematics kin, Vector3 tcp, Quaternion rot, float[] qInit,
                               out float[] best, out float bestPe, out float bestAe)
    {
        var seeds = new List<float[]> { (float[])qInit.Clone() };
        float[][] leans = { new[] { -60f, 100f, -40f }, new[] { -90f, 90f, -90f }, new[] { -120f, 120f, 0f } };
        for (int pan = -180; pan < 180; pan += 45)
            foreach (var l in leans)
                foreach (var w2 in new[] { 0f, 90f })
                {
                    var sd = (float[])qInit.Clone();
                    sd[0] = pan * Mathf.Deg2Rad;
                    if (sd.Length > 3) { sd[1] = l[0] * Mathf.Deg2Rad; sd[2] = l[1] * Mathf.Deg2Rad; sd[3] = l[2] * Mathf.Deg2Rad; }
                    if (sd.Length > 4) sd[4] = w2 * Mathf.Deg2Rad;
                    seeds.Add(sd);
                }

        best = null; bestPe = float.MaxValue; bestAe = float.MaxValue;
        float bestCost = float.MaxValue;
        foreach (var sd in seeds)
        {
            bool ok = kin.SolveIK(tcp, rot, sd, out var q, out float pe, out float ae);
            if (!ok)
            {
                if (best == null && pe < bestPe) { bestPe = pe; bestAe = ae; }
                continue;
            }
            float cost = 0f;
            for (int j = 0; j < q.Length; j++) cost += Mathf.Abs(q[j] - qInit[j]);
            if (cost < bestCost) { bestCost = cost; best = q; bestPe = pe; bestAe = ae; }
        }
        return best != null;
    }
}
