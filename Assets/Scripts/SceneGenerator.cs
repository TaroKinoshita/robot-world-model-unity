using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using UnityEngine;

/// <summary>P4-B: seed からシーン(ターゲット・ゴール・円柱)と候補を作る設定</summary>
[Serializable]
public class AutoSceneSettings
{
    [Tooltip("on: seed からターゲットの位置・向き、ゴール、候補、円柱の位置を作る(candidateSpecs は使わない)")]
    public bool enabled = false;

    [Header("Target")]
    public Vector2 targetXRange = new Vector2(-0.15f, 0.15f);
    public Vector2 targetZRange = new Vector2(0.22f, 0.42f);
    [Tooltip("鉛直軸まわりの向き(度)。立方体なので 0〜90° で全部の向きになる")]
    public Vector2 targetYawRangeDeg = new Vector2(0f, 90f);

    [Header("Goal(半径は EpisodeRecorder.goalRadius)")]
    [Tooltip("ターゲットの中心からゴールの中心までの距離(m)")]
    public Vector2 goalDistanceRange = new Vector2(0.06f, 0.12f);

    [Header("Candidates")]
    public int numMatchedPairs = 2;
    public int numSingles = 12;
    [Tooltip("単独候補のうち、ゴールの方向の近くに振る数(goal の True / False を半々に近づける)")]
    public int numGoalDirected = 6;
    [Tooltip("ゴールの方向 = ターゲットの面の法線(どれか 1 面)± この角度。斜めに押すと箱が回って横へそれるため")]
    public float goalFaceJitterDeg = 10f;
    [Tooltip("ゴール方向の候補の押す距離 = ゴールまでの距離 + これ ± goalDirectedLengthSpread(押し始めの隙間などで、実際の移動は押す距離より短い)")]
    public float goalLengthBias = 0.015f;
    public float goalDirectedAngleSpreadDeg = 10f;
    [Tooltip("単独候補のうち、ターゲットを円柱の方へ押す数(Task B でターゲット → 円柱の衝突を作る)。円柱を置いたあとで作る")]
    public int numTowardSecondary = 3;
    [Tooltip("円柱方向の候補の押す距離 = 円柱に届く距離 + この範囲(m)")]
    public Vector2 towardSecondaryOvershoot = new Vector2(0.03f, 0.06f);
    public float towardSecondaryAngleSpreadDeg = 10f;
    [Tooltip("円柱の置き場所を選ぶとき、ターゲットから 15 cm より遠い分に掛ける減点(m あたり)")]
    public float secondaryNearTargetWeight = 0.02f;
    public float goalDirectedLengthSpread = 0.03f;
    [Tooltip("押す距離(m)。ゴール方向の候補は、ゴールまでの距離 ± goalDirectedLengthSpread")]
    public Vector2 pushLengthRange = new Vector2(0.05f, 0.12f);
    [Tooltip("押し始め(pre_contact)が土台(根元)からこれより近い候補は外す(m)")]
    public float minPreContactDistFromBase = 0.25f;
    [Tooltip("1 step の関節変化がこれより大きい候補は外す(度)")]
    public float maxJointStepDeg = 2f;
    [Tooltip("matched pair の wrist-up の開始の枝を選ぶ初期値(度)")]
    public float[] wristUpSeedDeg = { -64f, -122f, 117f, 95f, 90f, 116f };

    [Header("Table / view")]
    [Tooltip("机の天板の範囲(x, z の最小と大きさ)")]
    public Rect tableXZ = new Rect(-0.5f, -0.15f, 1.0f, 0.8f);
    [Tooltip("物体・ゴール・押した後の位置を、机の端からどれだけ離すか(m)")]
    public float tableMargin = 0.06f;
    [Tooltip("画面の端からどれだけ内側に写るか(画面幅 = 1)")]
    public float viewMargin = 0.05f;

    [Header("Secondary(Task B の円柱)")]
    public float secondaryRadius = 0.048f;
    [Tooltip("円柱の高さの候補(m)。低い方から試して、片方だけ当たる置き場所が見つかった高さを使う")]
    public Vector2 secondaryHeightRange = new Vector2(0.24f, 0.40f);
    public float secondaryHeightStep = 0.02f;
    [Tooltip("matched pair の片方の腕が円柱に重なる深さの範囲(m)")]
    public Vector2 armOverlapRange = new Vector2(0.003f, 0.010f);
    [Tooltip("もう片方の腕と円柱の隙間(m)")]
    public float otherBranchClearance = 0.03f;
    [Tooltip("手(tool0 から先)と円柱の隙間、両方の枝(m)")]
    public float handClearance = 0.025f;
    [Tooltip("matched pair のターゲットの通り道・初期位置と円柱の隙間(m)")]
    public float targetPathClearance = 0.03f;
    [Tooltip("ゴールの中心と円柱の中心の距離の下限(m)")]
    public float goalClearance = 0.06f;
    public float gridStep = 0.01f;

    public int maxSceneAttempts = 20;
    public int maxCandidateAttempts = 300;
}

/// <summary>P4-B: 作ったシーンの中身(scene_generation.json に書く)</summary>
public class GeneratedScene
{
    public bool ok;
    public string error;
    public int seed;
    public Vector3 targetPos;
    public Quaternion targetRot;
    public float targetYawDeg;
    public Vector3 goal;
    public float goalDirDeg, goalDist;
    public List<CandidateSpec> specs = new List<CandidateSpec>();
    public Vector3 secondaryPos;
    public float secondaryHeight;
    public string placementPair, overlapBranch;
    public float armOverlap, otherClearance, handClear;
    public int sceneAttempts, plannedCandidates, rejectedCandidates;
    public Dictionary<string, int> rejectReasons = new Dictionary<string, int>();
    public List<string> placementLog = new List<string>();
}

/// <summary>P4-B: seed からシーンと候補を作る</summary>
public static class SceneGenerator
{
    public class Context
    {
        public ArmKinematics kin;
        public float[] qInit;
        public PlannerSettings planner;
        public float dt;
        public Vector3 targetHalf;      // 箱の座標での半分の大きさ
        public float targetCenterY;
        public float tableTop;
        public Camera cam;
        public Transform robotRoot;
        public string toolLinkName;
    }

    static float U(System.Random r, float a, float b) => a + (float)r.NextDouble() * (b - a);
    static float U(System.Random r, Vector2 range) => U(r, range.x, range.y);
    static Vector3 Dir(float deg) { float t = deg * Mathf.Deg2Rad; return new Vector3(Mathf.Sin(t), 0f, Mathf.Cos(t)); }

    static void Reject(GeneratedScene g, string why) { g.rejectedCandidates++; g.rejectReasons[why] = (g.rejectReasons.TryGetValue(why, out var n) ? n : 0) + 1; }

    static bool OnTable(AutoSceneSettings s, Vector3 p, float extra)
    {
        float m = s.tableMargin + extra;
        return p.x >= s.tableXZ.xMin + m && p.x <= s.tableXZ.xMax - m && p.z >= s.tableXZ.yMin + m && p.z <= s.tableXZ.yMax - m;
    }

    static bool InView(Context c, AutoSceneSettings s, Vector3 center, Vector3 half, Quaternion rot)
    {
        for (int i = 0; i < 8; i++)
        {
            var o = new Vector3((i & 1) == 0 ? -half.x : half.x, (i & 2) == 0 ? -half.y : half.y, (i & 4) == 0 ? -half.z : half.z);
            var v = c.cam.WorldToViewportPoint(center + rot * o);
            if (v.z <= 0f || v.x < s.viewMargin || v.x > 1f - s.viewMargin || v.y < s.viewMargin || v.y > 1f - s.viewMargin) return false;
        }
        return true;
    }

    static PlannedTrajectory PlanSpec(Context c, Vector3 tPos, Quaternion tRot, CandidateSpec spec)
    {
        var center = new Vector3(tPos.x, c.targetCenterY, tPos.z);
        return TrajectoryPlanner.Plan(c.kin, c.qInit, center, c.targetHalf, tRot, spec, c.planner, c.dt);
    }

    // 計画できて、品質の条件を満たすか
    static bool Feasible(Context c, AutoSceneSettings s, GeneratedScene g, Vector3 tPos, Quaternion tRot, CandidateSpec spec, out PlannedTrajectory tr)
    {
        tr = null;
        Vector3 u = Dir(spec.pushAngleDeg);
        // 押し始めが土台に近すぎる方向は IK を解く前に外す(押し始めは中心から手前へ 10 cm 程度)
        Vector3 approxPre = tPos - u * (TrajectoryPlanner.Support(u, c.targetHalf, tRot) + 0.06f);
        if (new Vector2(approxPre.x, approxPre.z).magnitude < s.minPreContactDistFromBase) { Reject(g, "near_base"); return false; }
        Vector3 tEnd = tPos + u * spec.pushLength;
        if (!OnTable(s, tEnd, 0f)) { Reject(g, "target_end_off_table"); return false; }
        g.plannedCandidates++;
        tr = PlanSpec(c, tPos, tRot, spec);
        if (!tr.ok) { Reject(g, "ik_failed"); return false; }
        if (tr.maxJointStep > s.maxJointStepDeg * Mathf.Deg2Rad) { Reject(g, "joint_step"); return false; }
        var pre = tr.waypoints[1];
        if (new Vector2(pre.x, pre.z).magnitude < s.minPreContactDistFromBase) { Reject(g, "near_base"); return false; }
        return true;
    }

    public static GeneratedScene Generate(Context c, AutoSceneSettings s, int seed)
    {
        var g = new GeneratedScene { seed = seed };
        var rng = new System.Random(seed);
        var trajs = new Dictionary<CandidateSpec, PlannedTrajectory>();

        for (int sa = 0; sa < s.maxSceneAttempts; sa++)
        {
            g.sceneAttempts = sa + 1;
            g.specs.Clear(); trajs.Clear();

            // ---- ターゲット ----
            float yaw = U(rng, s.targetYawRangeDeg);
            var tRot = Quaternion.Euler(0f, yaw, 0f);
            var tPos = new Vector3(U(rng, s.targetXRange), c.targetCenterY, U(rng, s.targetZRange));
            if (!OnTable(s, tPos, 0f) || !InView(c, s, tPos, c.targetHalf, tRot)) continue;

            // ---- ゴール(押せる方向に置く) ----
            bool goalOk = false;
            float gDir = 0f, gDist = 0f; Vector3 goal = Vector3.zero;
            for (int ga = 0; ga < 40 && !goalOk; ga++)
            {
                gDir = yaw + 90f * rng.Next(4) + U(rng, -s.goalFaceJitterDeg, s.goalFaceJitterDeg);
                gDir = Mathf.Repeat(gDir + 180f, 360f) - 180f;
                gDist = U(rng, s.goalDistanceRange);
                goal = tPos + Dir(gDir) * gDist;
                if (!OnTable(s, goal, 0f) || !InView(c, s, goal, c.targetHalf, tRot)) continue;
                var probe = new CandidateSpec { pushAngleDeg = gDir, pushLength = gDist + s.goalLengthBias };
                goalOk = Feasible(c, s, g, tPos, tRot, probe, out _);
            }
            if (!goalOk) continue;

            // ---- matched pair(手首の反転。だめなら 180° flip) ----
            int attempts = 0;
            for (int p = 0; p < s.numMatchedPairs && attempts < s.maxCandidateAttempts; )
            {
                attempts++;
                float dir = U(rng, -180f, 180f), len = U(rng, s.pushLengthRange);
                bool added = false;
                foreach (bool flip in new[] { false, true })
                {
                    string mp = $"mp{p}";
                    var down = new CandidateSpec { note = $"matched pair {mp}: wrist-down", pushAngleDeg = dir, pushLength = len, cartesianTransfer = true, matchedPairId = mp, branchLabel = "wrist_down", gripperYawFlip = flip };
                    var up = new CandidateSpec { note = $"matched pair {mp}: wrist-up, same TCP path", pushAngleDeg = dir, pushLength = len, cartesianTransfer = true, matchedPairId = mp, branchLabel = "wrist_up", gripperYawFlip = flip, startBranchSeedDeg = (float[])s.wristUpSeedDeg.Clone() };
                    if (!Feasible(c, s, g, tPos, tRot, down, out var trD)) continue;
                    if (!Feasible(c, s, g, tPos, tRot, up, out var trU)) continue;
                    g.specs.Add(down); g.specs.Add(up); trajs[down] = trD; trajs[up] = trU;
                    added = true; break;
                }
                if (added) p++;
            }
            if (g.specs.Count < 2 * s.numMatchedPairs) continue;

            // ---- 単独候補(前半はゴールの方向の近く、後半は全方向) ----
            int singles = 0;
            attempts = 0;
            int nBefore = s.numSingles - s.numTowardSecondary;   // 円柱方向の候補は、円柱を置いたあとで作る
            while (singles < nBefore && attempts < s.maxCandidateAttempts)
            {
                attempts++;
                bool toGoal = singles < s.numGoalDirected;
                float dir = toGoal ? gDir + U(rng, -s.goalDirectedAngleSpreadDeg, s.goalDirectedAngleSpreadDeg) : U(rng, -180f, 180f);
                float len = toGoal ? Mathf.Clamp(gDist + s.goalLengthBias + U(rng, -s.goalDirectedLengthSpread, s.goalDirectedLengthSpread), 0.03f, 0.15f) : U(rng, s.pushLengthRange);
                var spec = new CandidateSpec { note = toGoal ? "single, toward the goal" : "single, random direction", pushAngleDeg = dir, pushLength = len };
                if (!Feasible(c, s, g, tPos, tRot, spec, out var tr)) continue;
                g.specs.Add(spec); trajs[spec] = tr; singles++;
            }
            if (singles < nBefore) continue;

            // ---- 円柱(matched pair の片方の腕だけが当たる位置) ----
            bool placed = s.numMatchedPairs == 0;   // matched pair が無いときは円柱を動かさない(テスト用)
            for (int p = 0; p < s.numMatchedPairs && !placed; p++)
            {
                var down = g.specs[2 * p]; var up = g.specs[2 * p + 1];
                placed = PlaceSecondary(c, s, g, tPos, tRot, goal, down, up, trajs[down], trajs[up]);
                if (placed) g.placementPair = down.matchedPairId;
            }
            if (!placed) continue;

            // ---- 単独候補(円柱の方向へ押す。届かなければ数を満たすまでランダム方向で埋める) ----
            if (s.numTowardSecondary > 0)
            {
                Vector3 toCyl = new Vector3(g.secondaryPos.x - tPos.x, 0f, g.secondaryPos.z - tPos.z);
                float cylDir = Mathf.Atan2(toCyl.x, toCyl.z) * Mathf.Rad2Deg;
                int added = 0; attempts = 0;
                while (added < s.numTowardSecondary && attempts < s.maxCandidateAttempts)
                {
                    attempts++;
                    bool toward = attempts <= s.maxCandidateAttempts / 2;
                    float dir = toward ? cylDir + U(rng, -s.towardSecondaryAngleSpreadDeg, s.towardSecondaryAngleSpreadDeg) : U(rng, -180f, 180f);
                    Vector3 u = Dir(dir);
                    float reach = toCyl.magnitude - s.secondaryRadius - TrajectoryPlanner.Support(u, c.targetHalf, tRot);
                    float len = toward ? reach + U(rng, s.towardSecondaryOvershoot) : U(rng, s.pushLengthRange);
                    if (len < 0.03f || len > 0.20f) { if (toward) { Reject(g, "secondary_too_far"); continue; } }
                    var spec = new CandidateSpec { note = toward ? "single, toward the secondary object" : "single, random direction (no toward-secondary push was feasible)", pushAngleDeg = dir, pushLength = len };
                    if (!Feasible(c, s, g, tPos, tRot, spec, out var tr)) continue;
                    g.specs.Add(spec); trajs[spec] = tr; added++;
                }
                if (added < s.numTowardSecondary) continue;
            }

            g.targetPos = tPos; g.targetRot = tRot; g.targetYawDeg = yaw;
            g.goal = new Vector3(goal.x, 0f, goal.z); g.goalDirDeg = gDir; g.goalDist = gDist;
            g.ok = true;
            return g;
        }
        g.error = $"{s.maxSceneAttempts} 回試してもシーンを作れなかった。円柱: " + string.Join(" | ", g.placementLog.Take(10));
        return g;
    }

    // ---------- 円柱の置き場所 ----------

    class PointSet { public List<int> link = new List<int>(); public List<Vector3> local = new List<Vector3>(); }

    static PointSet armPts, handPts;   // arm: chain の各リンクの座標、hand: tool の座標

    static void CollectPoints(Context c)
    {
        if (armPts != null) return;
        armPts = new PointSet(); handPts = new PointSet();
        var names = c.kin.LinkNames;
        var linkT = names.Select(n => c.robotRoot.GetComponentsInChildren<ArticulationBody>(true).FirstOrDefault(a => a.name == n)).ToArray();
        var tool = c.kin.Tool;
        foreach (var col in c.robotRoot.GetComponentsInChildren<Collider>(false))
        {
            if (!col.enabled) continue;
            var owner = col.GetComponentInParent<ArticulationBody>();
            int li = Array.IndexOf(linkT, owner);
            bool isHand = li < 0 || owner.transform == tool || li == names.Length - 1 || names[li] == "wrist_3_link";
            var verts = new List<Vector3>();
            var mc = col as MeshCollider;
            if (mc != null && mc.sharedMesh != null) verts.AddRange(mc.sharedMesh.vertices.Select(v => col.transform.TransformPoint(v)));
            else { var b = col.bounds; for (int i = 0; i < 8; i++) verts.Add(b.center + Vector3.Scale(b.extents, new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1))); }
            int stride = Mathf.Max(1, verts.Count / 2000);
            for (int k = 0; k < verts.Count; k += stride)
            {
                if (isHand) { handPts.link.Add(-1); handPts.local.Add(tool.InverseTransformPoint(verts[k])); }
                else { armPts.link.Add(li); armPts.local.Add(linkT[li].transform.InverseTransformPoint(verts[k])); }
            }
        }
    }

    // 1 本の軌道について、xz の格子ごとに「その真上を通った点の一番低い高さ」を作る
    static float[,] MinHeightMap(Context c, AutoSceneSettings s, PlannedTrajectory tr, PointSet ps, bool hand)
    {
        int nx = Mathf.CeilToInt(s.tableXZ.width / s.gridStep) + 1, nz = Mathf.CeilToInt(s.tableXZ.height / s.gridStep) + 1;
        var map = new float[nx, nz];
        for (int i = 0; i < nx; i++) for (int j = 0; j < nz; j++) map[i, j] = float.PositiveInfinity;
        int L = c.kin.LinkNames.Length;
        var pos = new Vector3[L]; var rot = new Quaternion[L];
        for (int st = 0; st < tr.q.Count; st += 2)
        {
            Vector3 tp = Vector3.zero; Quaternion trt = Quaternion.identity;
            if (hand) c.kin.ToolPose(tr.q[st], out tp, out trt); else c.kin.LinkPoses(tr.q[st], pos, rot);
            for (int k = 0; k < ps.local.Count; k++)
            {
                Vector3 w = hand ? tp + trt * ps.local[k] : pos[ps.link[k]] + rot[ps.link[k]] * ps.local[k];
                int i = Mathf.RoundToInt((w.x - s.tableXZ.xMin) / s.gridStep), j = Mathf.RoundToInt((w.z - s.tableXZ.yMin) / s.gridStep);
                if (i < 0 || j < 0 || i >= nx || j >= nz) continue;
                if (w.y < map[i, j]) map[i, j] = w.y;
            }
        }
        return map;
    }

    // 円柱(中心 cx, cz)と、高さ top より下を通った点との距離。重なっていれば負(-重なりの深さ)
    static float Clearance(AutoSceneSettings s, float[,] map, float cx, float cz, float top, float search)
    {
        int nx = map.GetLength(0), nz = map.GetLength(1);
        int r = Mathf.CeilToInt((s.secondaryRadius + search) / s.gridStep);
        int ci = Mathf.RoundToInt((cx - s.tableXZ.xMin) / s.gridStep), cj = Mathf.RoundToInt((cz - s.tableXZ.yMin) / s.gridStep);
        float best = search;
        for (int i = Mathf.Max(0, ci - r); i <= Mathf.Min(nx - 1, ci + r); i++)
            for (int j = Mathf.Max(0, cj - r); j <= Mathf.Min(nz - 1, cj + r); j++)
            {
                if (map[i, j] >= top) continue;
                float x = s.tableXZ.xMin + i * s.gridStep, z = s.tableXZ.yMin + j * s.gridStep;
                float d = Mathf.Sqrt((x - cx) * (x - cx) + (z - cz) * (z - cz)) - s.secondaryRadius;
                if (d < best) best = d;
            }
        return best;
    }

    static float SegDist2D(Vector2 p, Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a; float t = Mathf.Clamp01(Vector2.Dot(p - a, ab) / Mathf.Max(1e-9f, ab.sqrMagnitude));
        return (p - (a + ab * t)).magnitude;
    }

    // 軌道の全 step(1 つおき)の、当たり判定の頂点の world 座標
    static List<Vector3> WorldPoints(Context c, PlannedTrajectory tr, PointSet ps, bool hand, float yMax)
    {
        var list = new List<Vector3>();
        int L = c.kin.LinkNames.Length;
        var pos = new Vector3[L]; var rot = new Quaternion[L];
        for (int st = 0; st < tr.q.Count; st += 2)
        {
            Vector3 tp = Vector3.zero; Quaternion trt = Quaternion.identity;
            if (hand) c.kin.ToolPose(tr.q[st], out tp, out trt); else c.kin.LinkPoses(tr.q[st], pos, rot);
            for (int k = 0; k < ps.local.Count; k++)
            {
                Vector3 w = hand ? tp + trt * ps.local[k] : pos[ps.link[k]] + rot[ps.link[k]] * ps.local[k];
                if (w.y <= yMax) list.Add(w);
            }
        }
        return list;
    }

    // カプセル(机の上に立てた円柱。下端 y0、高さ h、半径 r)の表面までの距離。中なら負
    static float CapsuleClearance(List<Vector3> pts, float cx, float cz, float y0, float h, float r)
    {
        return CapsuleClearance(pts, cx, cz, y0, h, r, out _);
    }

    static float CapsuleClearance(List<Vector3> pts, float cx, float cz, float y0, float h, float r, out Vector3 closest)
    {
        float ya = y0 + r, yb = y0 + h - r, best = float.PositiveInfinity;
        closest = Vector3.zero;
        for (int i = 0; i < pts.Count; i++)
        {
            var p = pts[i];
            float dy = p.y < ya ? p.y - ya : (p.y > yb ? p.y - yb : 0f);
            float dx = p.x - cx, dz = p.z - cz;
            float d2 = dx * dx + dz * dz + dy * dy;
            if (d2 < best) { best = d2; closest = p; }
        }
        return float.IsPositiveInfinity(best) ? best : Mathf.Sqrt(best) - r;
    }

    struct Cand { public float x, z, score; }

    static bool PlaceSecondary(Context c, AutoSceneSettings s, GeneratedScene g, Vector3 tPos, Quaternion tRot, Vector3 goal,
                               CandidateSpec down, CandidateSpec up, PlannedTrajectory trD, PlannedTrajectory trU)
    {
        CollectPoints(c);
        var maps = new[] { MinHeightMap(c, s, trD, armPts, false), MinHeightMap(c, s, trU, armPts, false),
                           MinHeightMap(c, s, trD, handPts, true), MinHeightMap(c, s, trU, handPts, true) };
        float yMax = c.tableTop + s.secondaryHeightRange.y + 0.01f;
        var lists = new[] { WorldPoints(c, trD, armPts, false, yMax), WorldPoints(c, trU, armPts, false, yMax),
                            WorldPoints(c, trD, handPts, true, yMax), WorldPoints(c, trU, handPts, true, yMax) };
        for (float H = s.secondaryHeightRange.x; H <= s.secondaryHeightRange.y + 1e-4f; H += s.secondaryHeightStep)
            if (PlaceSecondaryAt(c, s, g, tPos, tRot, goal, down, up, maps, lists, H)) { g.secondaryHeight = H; return true; }
        return false;
    }

    static bool PlaceSecondaryAt(Context c, AutoSceneSettings s, GeneratedScene g, Vector3 tPos, Quaternion tRot, Vector3 goal,
                                 CandidateSpec down, CandidateSpec up, float[][,] maps, List<Vector3>[] lists, float H)
    {
        float top = c.tableTop + H;
        var armD = maps[0]; var armU = maps[1]; var handD = maps[2]; var handU = maps[3];
        float tRad = new Vector2(c.targetHalf.x, c.targetHalf.z).magnitude;
        Vector3 u = Dir(down.pushAngleDeg);
        Vector2 a = new Vector2(tPos.x, tPos.z), b = a + new Vector2(u.x, u.z) * down.pushLength;
        float midOverlap = 0.5f * (s.armOverlapRange.x + s.armOverlapRange.y);
        float slack = s.gridStep;   // 格子の粗さの分だけ条件をゆるめて候補を集め、あとで正確に測る

        // ---- 1) 格子で候補を集める(近似) ----
        var cands = new List<Cand>();
        int gPath = 0, gGoal = 0, gHand = 0, gRange = 0, gView = 0;
        float gBestOv = float.NegativeInfinity;
        float m = s.tableMargin + s.secondaryRadius;
        for (float x = s.tableXZ.xMin + m; x <= s.tableXZ.xMax - m; x += s.gridStep)
            for (float z = s.tableXZ.yMin + m; z <= s.tableXZ.yMax - m; z += s.gridStep)
            {
                var p2 = new Vector2(x, z);
                if (SegDist2D(p2, a, b) < s.secondaryRadius + tRad + s.targetPathClearance) { gPath++; continue; }
                if ((p2 - new Vector2(goal.x, goal.z)).magnitude < s.goalClearance + s.secondaryRadius) { gGoal++; continue; }
                float h = Mathf.Min(Clearance(s, handD, x, z, top, 0.1f), Clearance(s, handU, x, z, top, 0.1f));
                if (h < s.handClearance - slack) { gHand++; continue; }
                float cD = Clearance(s, armD, x, z, top, 0.1f), cU = Clearance(s, armU, x, z, top, 0.1f);
                for (int k = 0; k < 2; k++)
                {
                    float ov = -(k == 0 ? cD : cU), other = k == 0 ? cU : cD;
                    if (other >= s.otherBranchClearance - slack) gBestOv = Mathf.Max(gBestOv, ov);
                    if (ov < s.armOverlapRange.x - slack || ov > s.armOverlapRange.y + slack || other < s.otherBranchClearance - slack) { gRange++; continue; }
                    var center = new Vector3(x, c.tableTop + 0.5f * H, z);
                    if (!InView(c, s, center, new Vector3(s.secondaryRadius, 0.5f * H, s.secondaryRadius), Quaternion.identity)) { gView++; continue; }
                    cands.Add(new Cand { x = x, z = z, score = Mathf.Abs(ov - midOverlap) });
                    break;
                }
            }
        if (cands.Count == 0) { g.placementLog.Add($"{down.matchedPairId} H{H * 100f:F0}cm: 格子の候補 0(通り道 {gPath}, ゴール {gGoal}, 手 {gHand}, 重なり {gRange}, 画面 {gView}, 片方だけの最大の重なり {gBestOv * 1000f:F0} mm)"); return false; }

        // ---- 2) 良さそうな順に、頂点とカプセルの距離で正確に測る ----
        var wArmD = lists[0]; var wArmU = lists[1]; var wHandD = lists[2]; var wHandU = lists[3];
        float bestScore = float.PositiveInfinity; bool found = false;
        int nHand = 0, nRange = 0;
        float closest = float.PositiveInfinity; string closestInfo = "";
        foreach (var cd in cands.OrderBy(q => q.score).Take(24))
        {
            for (int k = 0; k < 2; k++)
            {
                // 重なる側の腕の一番近い点へ向かって、重なりが目標になるまで中心をずらす(数回)
                var hitPts = k == 0 ? wArmD : wArmU; var otherPts = k == 0 ? wArmU : wArmD;
                float cx = cd.x, cz = cd.z, ov = 0f;
                for (int it = 0; it < 4; it++)
                {
                    float cl = CapsuleClearance(hitPts, cx, cz, c.tableTop, H, s.secondaryRadius, out var cp);
                    if (float.IsPositiveInfinity(cl)) break;
                    ov = -cl;
                    if (Mathf.Abs(ov - midOverlap) < 0.0005f) break;
                    var dir = new Vector2(cp.x - cx, cp.z - cz);
                    if (dir.sqrMagnitude < 1e-10f) break;
                    dir.Normalize();
                    float step = Mathf.Clamp(midOverlap - ov, -0.03f, 0.03f);
                    cx += dir.x * step; cz += dir.y * step;
                }
                ov = -CapsuleClearance(hitPts, cx, cz, c.tableTop, H, s.secondaryRadius);
                float other = CapsuleClearance(otherPts, cx, cz, c.tableTop, H, s.secondaryRadius);
                float h = Mathf.Min(CapsuleClearance(wHandD, cx, cz, c.tableTop, H, s.secondaryRadius),
                                    CapsuleClearance(wHandU, cx, cz, c.tableTop, H, s.secondaryRadius));
                float miss = Mathf.Max(0f, s.armOverlapRange.x - ov) + Mathf.Max(0f, ov - s.armOverlapRange.y) + Mathf.Max(0f, s.otherBranchClearance - other) + Mathf.Max(0f, s.handClearance - h);
                if (miss < closest) { closest = miss; closestInfo = $"ov {ov * 1000f:F1} mm, other {other * 1000f:F1} mm, hand {h * 1000f:F1} mm"; }
                if (h < s.handClearance) { nHand++; continue; }
                if (ov < s.armOverlapRange.x || ov > s.armOverlapRange.y || other < s.otherBranchClearance) { nRange++; continue; }
                // ずらした後の位置でも、机・画面・ターゲットの通り道・ゴールの条件を確かめる
                var p2 = new Vector2(cx, cz);
                if (cx < s.tableXZ.xMin + m || cx > s.tableXZ.xMax - m || cz < s.tableXZ.yMin + m || cz > s.tableXZ.yMax - m) continue;
                if (SegDist2D(p2, a, b) < s.secondaryRadius + tRad + s.targetPathClearance) continue;
                if ((p2 - new Vector2(goal.x, goal.z)).magnitude < s.goalClearance + s.secondaryRadius) continue;
                var center = new Vector3(cx, c.tableTop + 0.5f * H, cz);
                if (!InView(c, s, center, new Vector3(s.secondaryRadius, 0.5f * H, s.secondaryRadius), Quaternion.identity)) continue;
                // ターゲットに近い場所を少し優先する(ターゲットを円柱の方へ押す候補が届くように)。15 cm より遠いと 10 cm ごとに 2 mm 分の減点
                float tdist = new Vector2(cx - tPos.x, cz - tPos.z).magnitude;
                float score = Mathf.Abs(ov - midOverlap) + s.secondaryNearTargetWeight * Mathf.Max(0f, tdist - 0.15f);
                if (score < bestScore)
                {
                    bestScore = score; found = true;
                    g.secondaryPos = center;
                    g.overlapBranch = k == 0 ? down.branchLabel : up.branchLabel; g.armOverlap = ov; g.otherClearance = other; g.handClear = h;
                }
            }
            if (found && bestScore < 0.0005f) break;
        }
        if (!found) g.placementLog.Add($"{down.matchedPairId} H{H * 100f:F0}cm: 格子 {cands.Count}, 正確に測って 手が近い {nHand} / 重なりが範囲外 {nRange}, 一番近いもの: {closestInfo}");
        return found;
    }

    // ---------- 書き出し ----------

    static string N(float v) => v.ToString("G9", CultureInfo.InvariantCulture);
    static string V(Vector3 v) => $"[{N(v.x)}, {N(v.y)}, {N(v.z)}]";

    public static string ToJson(GeneratedScene g, AutoSceneSettings s, string sceneId)
    {
        var sb = new StringBuilder("{\n");
        sb.Append("  \"schema\": \"scene_generation_v1\",\n");
        sb.Append($"  \"scene_id\": \"{sceneId}\",\n");
        sb.Append($"  \"seed\": {g.seed},\n");
        sb.Append($"  \"ok\": {(g.ok ? "true" : "false")},\n");
        if (!g.ok) sb.Append($"  \"error\": \"{g.error}\",\n");
        sb.Append($"  \"target\": {{\"position_world\": {V(g.targetPos)}, \"yaw_deg\": {N(g.targetYawDeg)}}},\n");
        sb.Append($"  \"goal\": {{\"center_world\": {V(g.goal)}, \"direction_deg\": {N(g.goalDirDeg)}, \"distance_m\": {N(g.goalDist)}}},\n");
        sb.Append($"  \"secondary\": {{\"position_world\": {V(g.secondaryPos)}, \"radius_m\": {N(s.secondaryRadius)}, \"height_m\": {N(g.secondaryHeight)}, " +
                  $"\"placed_for_pair\": \"{g.placementPair}\", \"overlapping_branch\": \"{g.overlapBranch}\", \"arm_overlap_m\": {N(g.armOverlap)}, " +
                  $"\"other_branch_clearance_m\": {N(g.otherClearance)}, \"hand_clearance_m\": {N(g.handClear)}, " +
                  "\"method\": \"planned link poses x collider vertices (every 2nd step); xz grid to pre-select, then exact vertex-to-capsule distance\"},\n");
        sb.Append($"  \"attempts\": {{\"scene\": {g.sceneAttempts}, \"planned_candidates\": {g.plannedCandidates}, \"rejected_candidates\": {g.rejectedCandidates}, " +
                  $"\"reject_reasons\": {{{string.Join(", ", g.rejectReasons.Select(kv => $"\"{kv.Key}\": {kv.Value}"))}}}}},\n");
        sb.Append($"  \"settings\": {JsonUtility.ToJson(s)}\n");
        sb.Append("}\n");
        return sb.ToString();
    }
}
