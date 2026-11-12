using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

/// <summary>
/// 2 つ目のタスクの試作（物理）。Task2Pilot シーンだけで使う。本番のデータ生成とは無関係。
/// りんごを手先に固定した状態で区画の手前から始め、かごの上まで手先を直線で動かして離す。
/// 腕・手先が棚に触れたら、その場で止める（保護停止）。肩の左右の 2 つの解（A / B）で同じ手先の経路を回す。
/// 結果は Console（[T2P]）と Docs/task2/pilot_physics.json に書く。
/// </summary>
public class Task2PhysicsTest : MonoBehaviour
{
    [Serializable]
    public class Layout
    {
        public float basketX = 0f;      // かごの中心 x
        public float basketZ = 0.14f;   // 区画の手前の面からかごの中心までの奥行き
        public float width = 0.32f;     // 区画の内寸
        public float height = 0.50f;
        public float offset = 0f;       // 区画の中心 − かごの中心（x）
    }

    public bool autoRun = true;
    public float frontZ = 0.40f, depth = 0.32f, board = 0.02f, tableTop = 0.773f;
    public float approachDist = 0.30f, speed = 0.10f, holdAbove = 0.09f;
    public float settleSeconds = 2.0f;
    public float stiffness = 10000f, damping = 100f;
    public float[] seedADeg = { -13f, -72f, 98f, -117f, -90f, 0f };
    public float[] seedBDeg = { -167f, -108f, -98f, -63f, 90f, 0f };
    public List<Layout> layouts = new List<Layout>
    {
        new Layout { basketX = -0.05f, offset = -0.06f },
        new Layout { basketX = -0.05f, offset = 0.06f },
        new Layout { basketX = 0.05f, offset = -0.06f },
        new Layout { basketX = 0.05f, offset = 0.06f },
        new Layout { basketX = 0f, offset = 0f, width = 0.40f },
    };

    EpisodeRecorder rec;
    ArmKinematics kin;
    Transform shelf, basket, apple;
    Rigidbody appleRb;
    List<(Collider col, string link, bool tool)> robotCols;
    List<Collider> shelfCols, basketCols;
    readonly List<string> results = new List<string>();

    void Start() { if (autoRun) StartCoroutine(RunAll()); }

    IEnumerator RunAll()
    {
        for (int i = 0; i < 5; i++) yield return null;
        rec = FindObjectOfType<EpisodeRecorder>(true);
        kin = ArmKinematics.Build(rec.robotRoot, rec.toolLinkName, rec.armJointNames, rec.fingerPadA, rec.fingerPadB, out string err);
        if (kin == null) { Debug.LogError("[T2P] kinematics: " + err); yield break; }
        shelf = GameObject.Find("Task2Shelf/Shelf").transform;
        basket = shelf.Find("Basket");
        apple = GameObject.Find("Task2Shelf/Apple").transform;
        appleRb = apple.GetComponent<Rigidbody>();
        appleRb.interpolation = RigidbodyInterpolation.None;
        appleRb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        var gripRoot = rec.robotRoot.GetComponentsInChildren<Transform>(true).FirstOrDefault(t => t.name == rec.gripperRootName);
        robotCols = new List<(Collider, string, bool)>();
        foreach (var c in rec.robotRoot.GetComponentsInChildren<Collider>(false))
        {
            if (!c.enabled) continue;
            var ab = c.GetComponentInParent<ArticulationBody>();
            string ln = ab != null ? ab.name : c.name;
            if (ln == "world" || ln == "base_link") continue;
            bool tool = (gripRoot != null && c.transform.IsChildOf(gripRoot)) || ln == "tool0";
            robotCols.Add((c, ln, tool));
            Physics.IgnoreCollision(c, apple.GetComponent<Collider>(), tool);   // りんごとグリッパーは当てない
        }
        foreach (var ab in rec.robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (Array.IndexOf(kin.JointBodies, ab) >= 0) { var d = ab.xDrive; d.stiffness = stiffness; d.damping = damping; ab.xDrive = d; }
        toolLinks = new HashSet<string>(robotCols.Where(r => r.tool).Select(r => r.link));
        foreach (var c in shelf.GetComponentsInChildren<Collider>(true))
            if (c.GetComponent<ShelfContactProbe>() == null) c.gameObject.AddComponent<ShelfContactProbe>();
        ShelfContactProbe.Enabled = true;

        int id = 0;
        foreach (var L in layouts)
            foreach (var branch in new[] { "A", "B" })
            {
                yield return RunOne(id++, L, branch);
            }
        var json = "{\n  \"fixed_dt\": " + Time.fixedDeltaTime.ToString("G6") + ",\n  \"episodes\": [\n" + string.Join(",\n", results) + "\n  ]\n}\n";
        var path = Path.Combine(Application.dataPath, "..", "Docs", "task2", "pilot_physics.json");
        File.WriteAllText(path, json);
        Debug.Log("[T2P] DONE " + results.Count + " episodes -> " + path);
    }

    void ApplyLayout(Layout L)
    {
        float W = L.width, H = L.height, D = depth, th = board;
        float cx = L.basketX + L.offset;
        shelf.position = new Vector3(cx, tableTop, frontZ);
        void Set(string n, Vector3 c, Vector3 s) { var t = shelf.Find(n); t.localPosition = c; t.localScale = s; }
        Set("Bottom", new Vector3(0, th / 2, D / 2), new Vector3(W + 2 * th, th, D));
        Set("Top", new Vector3(0, th + H + th / 2, D / 2), new Vector3(W + 2 * th, th, D));
        Set("Left", new Vector3(-W / 2 - th / 2, th + H / 2, D / 2), new Vector3(th, H, D));
        Set("Right", new Vector3(W / 2 + th / 2, th + H / 2, D / 2), new Vector3(th, H, D));
        if (shelf.Find("Back").gameObject.activeSelf) Set("Back", new Vector3(0, th + H / 2 + th / 2, D + th / 2), new Vector3(W + 2 * th, H + 2 * th, th));
        basket.localPosition = new Vector3(-L.offset, th, L.basketZ);
        Physics.SyncTransforms();
        shelfCols = new List<Collider>();
        foreach (var n in new[] { "Bottom", "Top", "Left", "Right", "Back" }) if (shelf.Find(n).gameObject.activeSelf) shelfCols.Add(shelf.Find(n).GetComponent<Collider>());
        basketCols = basket.GetComponentsInChildren<Collider>().ToList();
    }

    IEnumerator RunOne(int id, Layout L, string branch)
    {
        ApplyLayout(L);
        Quaternion rot = kin.ToolRotationFor(Vector3.down, Vector3.right);
        // 下向き: 今の初期姿勢と同じ手先の向きを使う
        var rip = FindObjectOfType<RobotInitialPose>();
        kin.TcpPose(rip.jointDegrees.Select(d => d * Mathf.Deg2Rad).ToArray(), out _, out rot);

        var P = new Vector3(L.basketX, tableTop + board + holdAbove, frontZ + L.basketZ);
        var front = P - new Vector3(0, 0, approachDist);
        var seed = (branch == "A" ? seedADeg : seedBDeg).Select(d => d * Mathf.Deg2Rad).ToArray();
        string res0 = null;
        if (!kin.SolveIK(front, rot, seed, out var q, out float pe, out float ae) || pe > 1e-3f) res0 = "ik_start_failed";
        var plan = new List<float[]> { q };
        int n = Mathf.Max(1, Mathf.CeilToInt(approachDist / speed / Time.fixedDeltaTime));
        for (int k = 1; k <= n && res0 == null; k++)
        {
            float a = (float)k / n; a = a * a * a * (10 - 15 * a + 6 * a * a);   // min-jerk
            if (!kin.SolveIK(Vector3.Lerp(front, P, a), rot, q, out var qn, out pe, out ae) || pe > 1e-3f) { res0 = "ik_path_failed"; break; }
            q = qn; plan.Add(q);
        }
        if (res0 != null) { Report(id, L, branch, res0, -1, "", "", false, Vector3.zero, false); yield break; }

        // 初期姿勢を直接置く
        appleRb.isKinematic = true;
        for (int j = 0; j < kin.Dof; j++)
        {
            var ab = kin.JointBodies[j];
            ab.jointPosition = new ArticulationReducedSpace(plan[0][j]);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
            var d = ab.xDrive; d.target = plan[0][j] * Mathf.Rad2Deg; d.targetVelocity = 0; ab.xDrive = d;
        }
        for (int i = 0; i < 10; i++) yield return new WaitForFixedUpdate();
        // りんごを TCP に固定
        kin.TcpPose(ReadQ(), out var tcp, out _);
        apple.position = tcp; apple.rotation = Quaternion.identity;
        appleRb.isKinematic = false; appleRb.linearVelocity = Vector3.zero; appleRb.angularVelocity = Vector3.zero;
        var joint = apple.gameObject.AddComponent<FixedJoint>();
        joint.connectedArticulationBody = kin.JointBodies[kin.Dof - 1].GetComponentInChildren<ArticulationBody>();
        var toolAb = rec.robotRoot.GetComponentsInChildren<ArticulationBody>(true).FirstOrDefault(a => a.name == rec.toolLinkName) ?? kin.JointBodies[kin.Dof - 1];
        joint.connectedArticulationBody = toolAb;
        for (int i = 0; i < 5; i++) yield return new WaitForFixedUpdate();
        ShelfContactProbe.Hits.Clear();

        int stopStep = -1; var armHits = new SortedSet<string>(); var toolHits = new SortedSet<string>(); bool appleShelf = false;
        float maxTcpErr = 0f;
        for (int s = 1; s < plan.Count; s++)
        {
            for (int j = 0; j < kin.Dof; j++)
            {
                var ab = kin.JointBodies[j]; var d = ab.xDrive;
                d.target = plan[s][j] * Mathf.Rad2Deg;
                d.targetVelocity = (plan[s][j] - plan[s - 1][j]) / Time.fixedDeltaTime * Mathf.Rad2Deg;
                ab.xDrive = d;
            }
            yield return new WaitForFixedUpdate();
            kin.TcpPose(plan[s], out var tPlan, out _); kin.TcpPose(ReadQ(), out var tNow, out _);
            maxTcpErr = Mathf.Max(maxTcpErr, Vector3.Distance(tPlan, tNow));
            bool any = CheckContacts(armHits, toolHits, ref appleShelf);
            if (any)
            {
                stopStep = s;   // 保護停止: 今の姿勢で止める
                var qNow = ReadQ();
                for (int j = 0; j < kin.Dof; j++) { var ab = kin.JointBodies[j]; var d = ab.xDrive; d.target = qNow[j] * Mathf.Rad2Deg; d.targetVelocity = 0; ab.xDrive = d; }
                break;
            }
        }
        if (stopStep < 0)
        {
            for (int i = 0; i < 10; i++) yield return new WaitForFixedUpdate();
            Destroy(joint);   // 離す
        }
        else
        {
            for (int i = 0; i < 10; i++) { yield return new WaitForFixedUpdate(); CheckContacts(armHits, toolHits, ref appleShelf); }
        }
        float t = 0f;
        while (t < settleSeconds) { yield return new WaitForFixedUpdate(); t += Time.fixedDeltaTime; if (stopStep < 0) CheckContacts(armHits, toolHits, ref appleShelf); }
        // かごの中で止まったか
        var bb = basketCols[0].bounds; foreach (var bc in basketCols) bb.Encapsulate(bc.bounds); bb.Expand(new Vector3(0f, 0.06f, 0f));   // かごの当たり判定の箱を合わせた範囲（上に少し伸ばす）
        bool placed = joint == null && bb.Contains(apple.position) && appleRb.linearVelocity.magnitude < 0.02f;
        Report(id, L, branch, stopStep >= 0 ? "stopped" : "completed", stopStep, string.Join("|", armHits), string.Join("|", toolHits), appleShelf, apple.position, placed, maxTcpErr, plan.Count);
        if (joint != null) Destroy(joint);
        yield return new WaitForFixedUpdate();
        appleRb.isKinematic = true;
    }

    HashSet<string> toolLinks;

    bool CheckContacts(SortedSet<string> arm, SortedSet<string> tool, ref bool appleShelf)
    {
        bool any = false;
        // PhysX の接触通知（押し戻されてめり込みが残らない接触も拾う）
        foreach (var (link, board, sep, _) in ShelfContactProbe.Hits)
        {
            bool isBasket = board.StartsWith("B") && board.Length <= 6 && board != "Bottom" && board != "Back";
            bool isTool = toolLinks.Contains(link);
            if (isBasket && !isTool) { arm.Add(link + ">" + board); any = true; continue; }
            (isTool ? tool : arm).Add(link + ">" + board); any = true;
        }
        ShelfContactProbe.Hits.Clear();
        foreach (var (c, ln, isTool) in robotCols)
            foreach (var s in shelfCols.Concat(isTool ? Enumerable.Empty<Collider>() : basketCols))
            {
                if (!c.bounds.Intersects(s.bounds)) continue;
                if (Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, s, s.transform.position, s.transform.rotation, out _, out _))
                {
                    any = true; (isTool ? tool : arm).Add(ln + ">" + s.name);
                }
            }
        // 手先とかごの縁（手先の接触）
        foreach (var (c, ln, isTool) in robotCols)
        {
            if (!isTool) continue;
            foreach (var s in basketCols)
                if (c.bounds.Intersects(s.bounds) && Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, s, s.transform.position, s.transform.rotation, out _, out _))
                { any = true; tool.Add(ln + ">" + s.name); }
        }
        var ac = apple.GetComponent<Collider>();
        foreach (var s in shelfCols)
            if (ac.bounds.Intersects(s.bounds) && Physics.ComputePenetration(ac, apple.position, apple.rotation, s, s.transform.position, s.transform.rotation, out _, out _))
                appleShelf = true;
        return any;
    }

    float[] ReadQ()
    {
        var q = new float[kin.Dof];
        for (int j = 0; j < kin.Dof; j++) q[j] = kin.JointBodies[j].jointPosition[0];
        return q;
    }

    void Report(int id, Layout L, string branch, string outcome, int stopStep, string armHits, string toolHits, bool appleShelf, Vector3 applePos, bool placed, float maxTcpErr = 0f, int planSteps = 0)
    {
        string N(float v) => v.ToString("G5", System.Globalization.CultureInfo.InvariantCulture);
        string line = $"    {{\"id\": {id}, \"branch\": \"{branch}\", \"basket_x\": {N(L.basketX)}, \"basket_z\": {N(L.basketZ)}, \"width\": {N(L.width)}, \"height\": {N(L.height)}, \"offset\": {N(L.offset)}, " +
                      $"\"outcome\": \"{outcome}\", \"stop_step\": {stopStep}, \"plan_steps\": {planSteps}, \"arm_shelf\": \"{armHits}\", \"tool_contact\": \"{toolHits}\", \"apple_shelf\": {(appleShelf ? "true" : "false")}, " +
                      $"\"placed\": {(placed ? "true" : "false")}, \"apple_final\": [{N(applePos.x)}, {N(applePos.y)}, {N(applePos.z)}], \"max_tcp_err_m\": {N(maxTcpErr)}}}";
        results.Add(line);
        Debug.Log("[T2P] " + line.Trim());
    }
}
