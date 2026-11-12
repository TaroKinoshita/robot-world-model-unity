using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

/// <summary>
/// 2 つ目のタスク（把持済みの運搬・配置）のデータを作る。Task2Pilot シーンで Play すると、シーンを seed 順に作って書き出す。
/// 1 シーン = 棚の配置 1 つ。候補 = 区画の手前での入り方（dx）× 肩の左右の解（A / B）。A と B は同じ手先の経路（matched pair）。
/// 書き出す形は、1 つ目のタスクの episode フォルダに合わせた最小限（Learning/wm_cache_t2.py が読む）:
///   scene_XXXX/ camera.json, scene_initial.json, scene_generation.json, candidates.json, metadata.json,
///               initial/robot_free/rgb.png・depth.exr（ロボットとりんごを除いた初期画像）
///   scene_XXXX/candidates/cXXX/ actions/（ActionRepresentations と同じ）, planned_trajectory.json,
///               final/final_state.json・target_mask.png（りんごの最後の見え方）, executed/summary.json
/// 済んだシーン（metadata.json がある）は飛ばすので、止めても続きから再開できる。
/// </summary>
public class Task2Recorder : MonoBehaviour
{
    [Header("出力")]
    public string outputRoot = "Episodes/_task2_v1";   // プロジェクトからの相対
    public int startSeed = 0;
    public int sceneCount = 300;
    public float stopAfterHours = 0f;                  // 0 = 止めない
    public float timeScale = 8f;
    public bool autoRun = true;

    [Header("配置（一様乱数の範囲）")]
    public Vector2 basketXRange = new Vector2(-0.07f, 0.13f);
    public Vector2 basketZRange = new Vector2(0.13f, 0.20f);   // 区画の手前の面からかごの中心まで
    public Vector2 widthRange = new Vector2(0.28f, 0.46f);
    public Vector2 heightRange = new Vector2(0.48f, 0.60f);
    public Vector2 offsetRange = new Vector2(-0.07f, 0.07f);   // 区画の中心 − かごの中心（x）

    [Header("棚・経路")]
    public float frontZ = 0.40f, depth = 0.32f, board = 0.02f, tableTop = 0.773f;
    public float approachDist = 0.30f, speed = 0.10f, holdAbove = 0.13f;
    public float[] approachDx = { -0.10f, -0.05f, 0f, 0.05f };
    public float settleSeconds = 2.0f;
    public float stiffness = 10000f, damping = 100f;
    public float[] seedADeg = { -13f, -72f, 98f, -117f, -90f, 0f };
    public float[] seedBDeg = { -167f, -108f, -98f, -63f, 90f, 0f };
    public ActionRasterSettings raster = new ActionRasterSettings();

    EpisodeRecorder rec;
    ArmKinematics kin;
    RobotRasterizer rr;
    Camera cam;
    RenderTexture rt;
    Transform shelf, basket, apple;
    Rigidbody appleRb;
    List<(Collider col, string link, bool tool)> robotCols;
    HashSet<string> toolLinks;
    List<Collider> shelfCols, basketCols;
    Renderer[] robotRenderers;
    string root;
    static readonly CultureInfo IC = CultureInfo.InvariantCulture;

    struct Layout { public float basketX, basketZ, width, height, offset; }

    void Start() { if (autoRun) StartCoroutine(Run()); }

    IEnumerator Run()
    {
        for (int i = 0; i < 5; i++) yield return null;
        rec = FindObjectOfType<EpisodeRecorder>(true);
        kin = ArmKinematics.Build(rec.robotRoot, rec.toolLinkName, rec.armJointNames, rec.fingerPadA, rec.fingerPadB, out string err);
        if (kin == null) { Debug.LogError("[T2R] kinematics: " + err); yield break; }
        rr = RobotRasterizer.Build(kin, rec.robotRoot);
        cam = Camera.main;
        rt = new RenderTexture(256, 256, 24, RenderTextureFormat.ARGB32) { name = "T2R_RT" };
        cam.targetTexture = rt;   // pixelWidth / Height = 256（CameraExporter・ActionRepresentations と同じ解像度）
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
            Physics.IgnoreCollision(c, apple.GetComponent<Collider>(), tool);
        }
        toolLinks = new HashSet<string>(robotCols.Where(r => r.tool).Select(r => r.link));
        robotRenderers = rec.robotRoot.GetComponentsInChildren<Renderer>(true).Where(r => r.enabled).ToArray();
        foreach (var ab in rec.robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (Array.IndexOf(kin.JointBodies, ab) >= 0) { var d = ab.xDrive; d.stiffness = stiffness; d.damping = damping; ab.xDrive = d; }
        foreach (var c in shelf.GetComponentsInChildren<Collider>(true))
            if (c.GetComponent<ShelfContactProbe>() == null) c.gameObject.AddComponent<ShelfContactProbe>();
        ShelfContactProbe.Enabled = true;

        root = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputRoot));
        Directory.CreateDirectory(root);
        Time.timeScale = timeScale;
        var t0 = DateTime.Now;
        int done = 0, skipped = 0;
        Debug.Log($"[T2R] start: {root}, seeds {startSeed}..{startSeed + sceneCount - 1}, timeScale {timeScale}");
        for (int i = 0; i < sceneCount; i++)
        {
            if (stopAfterHours > 0 && (DateTime.Now - t0).TotalHours >= stopAfterHours) { Debug.Log("[T2R] stopAfterHours に達したので止める"); break; }
            int seed = startSeed + i;
            string sid = $"scene_{seed:D4}";
            string sd = Path.Combine(root, sid);
            if (File.Exists(Path.Combine(sd, "metadata.json"))) { skipped++; continue; }
            if (Directory.Exists(sd)) Directory.Delete(sd, true);   // 途中で止まったシーンは作り直す
            yield return RunScene(seed, sid, sd);
            done++;
            Debug.Log($"[T2R] {sid} done ({done} new, {skipped} skipped, {(DateTime.Now - t0).TotalMinutes:F1} min)");
        }
        Time.timeScale = 1f;
        cam.targetTexture = null;
        File.WriteAllText(Path.Combine(root, "_batch_done.txt"), $"{DateTime.Now:s} new {done} skipped {skipped}\n");
        Debug.Log($"[T2R] DONE new {done}, skipped {skipped}");
    }

    // ---------- 配置 ----------

    Layout Sample(System.Random r)
    {
        float U(Vector2 v) => (float)(v.x + r.NextDouble() * (v.y - v.x));
        var L = new Layout { basketX = U(basketXRange), basketZ = U(basketZRange), width = U(widthRange), height = U(heightRange), offset = U(offsetRange) };
        // かごが区画の中に収まるように（側板との間に 1 cm 以上）
        float bxHalf = 0.08f, room = L.width / 2 - bxHalf - 0.01f;
        L.offset = Mathf.Clamp(L.offset, -room, room);
        return L;
    }

    void ApplyLayout(Layout L)
    {
        float W = L.width, H = L.height, D = depth, th = board;
        shelf.position = new Vector3(L.basketX + L.offset, tableTop, frontZ);
        void Set(string n, Vector3 c, Vector3 s) { var t = shelf.Find(n); t.localPosition = c; t.localScale = s; }
        Set("Bottom", new Vector3(0, th / 2, D / 2), new Vector3(W + 2 * th, th, D));
        Set("Top", new Vector3(0, th + H + th / 2, D / 2), new Vector3(W + 2 * th, th, D));
        Set("Left", new Vector3(-W / 2 - th / 2, th + H / 2, D / 2), new Vector3(th, H, D));
        Set("Right", new Vector3(W / 2 + th / 2, th + H / 2, D / 2), new Vector3(th, H, D));
        basket.localPosition = new Vector3(-L.offset, th, L.basketZ);
        Physics.SyncTransforms();
        shelfCols = new List<Collider>();
        foreach (var n in new[] { "Bottom", "Top", "Left", "Right", "Back" })
            if (shelf.Find(n).gameObject.activeSelf) shelfCols.Add(shelf.Find(n).GetComponent<Collider>());
        basketCols = basket.GetComponentsInChildren<Collider>().ToList();
    }

    // ---------- 1 シーン ----------

    IEnumerator RunScene(int seed, string sid, string sd)
    {
        var rng = new System.Random(seed);
        var L = Sample(rng);
        ApplyLayout(L);
        Directory.CreateDirectory(sd);

        // 初期姿勢を見えない所へ（ロボットとりんごを除いた画像を撮る）
        appleRb.isKinematic = true;
        apple.position = new Vector3(0, -5, 0);
        SetRobotVisible(false);
        yield return new WaitForEndOfFrame();
        CaptureRgb(Path.Combine(sd, "initial", "robot_free", "rgb.png"));
        WriteSceneDepth(Path.Combine(sd, "initial", "robot_free", "depth.exr"));
        SetRobotVisible(true);
        var exporter = cam.GetComponent<CameraExporter>();
        if (exporter == null) exporter = cam.gameObject.AddComponent<CameraExporter>();
        exporter.targetCamera = cam;
        exporter.ExportToFile(Path.Combine(sd, "camera.json"));

        // 候補を作る（A・B とも IK が通る入り方だけ）
        Quaternion rot;
        var rip = FindObjectOfType<RobotInitialPose>();
        kin.TcpPose(rip.jointDegrees.Select(d => d * Mathf.Deg2Rad).ToArray(), out _, out rot);
        var P = new Vector3(L.basketX, tableTop + board + holdAbove, frontZ + L.basketZ);
        var cands = new List<(string cid, int mp, string branch, float dx, PlannedTrajectory tr)>();
        int mpId = 0;
        foreach (float dx in approachDx)
        {
            var trA = Plan(P, P + new Vector3(dx, 0, -approachDist), rot, seedADeg);
            var trB = Plan(P, P + new Vector3(dx, 0, -approachDist), rot, seedBDeg);
            if (trA == null || trB == null) continue;
            cands.Add(($"c{cands.Count:D3}", mpId, "shoulder_a", dx, trA));
            cands.Add(($"c{cands.Count:D3}", mpId, "shoulder_b", dx, trB));
            mpId++;
        }

        var cj = new StringBuilder("{\n  \"scene_id\": \"" + sid + "\",\n  \"candidates\": [\n");
        for (int k = 0; k < cands.Count; k++)
        {
            var c = cands[k];
            string cdir = Path.Combine(sd, "candidates", c.cid);
            Directory.CreateDirectory(cdir);
            ActionRepresentations.Write(kin, c.tr, cam, cdir, raster, sid, c.cid, rr);
            File.WriteAllText(Path.Combine(cdir, "planned_trajectory.json"), PlanJson(c.tr), new UTF8Encoding(false));
            yield return Execute(c.tr, cdir, L);
            cj.Append($"    {{\"candidate_id\": \"{c.cid}\", \"matched_pair_id\": \"mp{c.mp}\", \"branch\": \"{c.branch}\", \"approach_dx\": {N(c.dx)}, \"note\": \"matched pair {c.mp}: {c.branch}, same TCP path\"}}");
            cj.Append(k < cands.Count - 1 ? ",\n" : "\n");
        }
        cj.Append("  ]\n}\n");
        File.WriteAllText(Path.Combine(sd, "candidates.json"), cj.ToString(), new UTF8Encoding(false));

        var bc = basket.position;
        File.WriteAllText(Path.Combine(sd, "scene_generation.json"),
            "{\n" +
            $"  \"seed\": {seed},\n" +
            $"  \"layout\": {{\"basket_x\": {N(L.basketX)}, \"basket_z_from_front\": {N(L.basketZ)}, \"width\": {N(L.width)}, \"height\": {N(L.height)}, \"offset\": {N(L.offset)}, \"depth\": {N(depth)}, \"board\": {N(board)}, \"front_z\": {N(frontZ)}}},\n" +
            $"  \"goal\": {{\"center_world\": [{N(bc.x)}, {N(bc.y)}, {N(bc.z)}], \"what\": \"basket floor center\"}},\n" +
            $"  \"target\": {{\"position_world\": [{N(P.x)}, {N(P.y)}, {N(P.z)}], \"yaw_deg\": 0, \"what\": \"TCP (apple center) at release\"}},\n" +
            $"  \"secondary\": {{\"position_world\": [{N(shelf.position.x)}, {N(shelf.position.y)}, {N(shelf.position.z + depth / 2)}], \"height_m\": {N(L.height)}, \"what\": \"shelf compartment center (bottom)\"}},\n" +
            $"  \"candidates\": {cands.Count}\n}}\n", new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(sd, "scene_initial.json"),
            $"{{\n  \"seed\": {seed},\n  \"task\": {{\"task_variant\": \"T2\", \"pair_id\": \"{sid.Replace("scene_", "pair_")}\"}},\n  \"unity_version\": \"{Application.unityVersion}\"\n}}\n", new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(sd, "metadata.json"),
            $"{{\n  \"schema\": \"task2_scene_v1\",\n  \"scene_id\": \"{sid}\",\n  \"created\": \"{DateTime.Now:s}\",\n  \"fixed_dt\": {N(Time.fixedDeltaTime)},\n  \"time_scale\": {N(timeScale)},\n  \"candidates\": {cands.Count}\n}}\n", new UTF8Encoding(false));
    }

    PlannedTrajectory Plan(Vector3 P, Vector3 front, Quaternion rot, float[] seedDeg)
    {
        var seed = seedDeg.Select(d => d * Mathf.Deg2Rad).ToArray();
        if (!kin.SolveIK(front, rot, seed, out var q, out float pe, out float ae) || pe > 1e-3f || ae > 1e-2f) return null;
        float dt = Time.fixedDeltaTime;
        var tr = new PlannedTrajectory { dt = dt, ok = true };
        void Add(float[] qq, string ph)
        {
            kin.TcpPose(qq, out var tcp, out var tool);
            tr.t.Add(tr.q.Count * dt); tr.phase.Add(ph); tr.q.Add(qq); tr.tcp.Add(tcp); tr.toolRot.Add(tool);
        }
        Add(q, "start");
        int n = Mathf.Max(1, Mathf.CeilToInt(approachDist / speed / dt));
        for (int k = 1; k <= n; k++)
        {
            float a = (float)k / n; a = a * a * a * (10 - 15 * a + 6 * a * a);
            if (!kin.SolveIK(Vector3.Lerp(front, P, a), rot, q, out var qn, out pe, out ae) || pe > 1e-3f) return null;
            float mx = 0; for (int j = 0; j < qn.Length; j++) mx = Mathf.Max(mx, Mathf.Abs(qn[j] - q[j]));
            if (mx > 0.2f) return null;
            q = qn; Add(q, "approach");
        }
        for (int k = 0; k < 10; k++) Add((float[])q.Clone(), "release");
        return tr;
    }

    // ---------- 実行（保護停止） ----------

    IEnumerator Execute(PlannedTrajectory tr, string cdir, Layout L)
    {
        appleRb.isKinematic = true;
        for (int j = 0; j < kin.Dof; j++)
        {
            var ab = kin.JointBodies[j];
            ab.jointPosition = new ArticulationReducedSpace(tr.q[0][j]);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
            var d = ab.xDrive; d.target = tr.q[0][j] * Mathf.Rad2Deg; d.targetVelocity = 0; ab.xDrive = d;
        }
        for (int i = 0; i < 10; i++) yield return new WaitForFixedUpdate();
        kin.TcpPose(ReadQ(), out var tcp0, out _);
        apple.position = tcp0; apple.rotation = Quaternion.identity;
        appleRb.isKinematic = false; appleRb.linearVelocity = Vector3.zero; appleRb.angularVelocity = Vector3.zero;
        var toolAb = rec.robotRoot.GetComponentsInChildren<ArticulationBody>(true).FirstOrDefault(a => a.name == rec.toolLinkName) ?? kin.JointBodies[kin.Dof - 1];
        var joint = apple.gameObject.AddComponent<FixedJoint>();
        joint.connectedArticulationBody = toolAb;
        for (int i = 0; i < 5; i++) yield return new WaitForFixedUpdate();
        ShelfContactProbe.Hits.Clear();

        int stopStep = -1; Vector3 firstContact = new Vector3(float.NaN, float.NaN, float.NaN);
        var armHits = new SortedSet<string>(); var toolHits = new SortedSet<string>(); bool appleShelf = false;
        float maxTcpErr = 0f;
        int nMove = tr.phase.FindLastIndex(p => p == "approach") + 1;
        for (int s = 1; s < nMove; s++)
        {
            for (int j = 0; j < kin.Dof; j++)
            {
                var ab = kin.JointBodies[j]; var d = ab.xDrive;
                d.target = tr.q[s][j] * Mathf.Rad2Deg;
                d.targetVelocity = (tr.q[s][j] - tr.q[s - 1][j]) / tr.dt * Mathf.Rad2Deg;
                ab.xDrive = d;
            }
            yield return new WaitForFixedUpdate();
            kin.TcpPose(ReadQ(), out var tNow, out _);
            maxTcpErr = Mathf.Max(maxTcpErr, Vector3.Distance(tr.tcp[s], tNow));
            if (CheckContacts(armHits, toolHits, ref appleShelf, ref firstContact))
            {
                stopStep = s;
                var qNow = ReadQ();
                for (int j = 0; j < kin.Dof; j++) { var ab = kin.JointBodies[j]; var d = ab.xDrive; d.target = qNow[j] * Mathf.Rad2Deg; d.targetVelocity = 0; ab.xDrive = d; }
                break;
            }
        }
        if (stopStep < 0)
        {
            for (int i = 0; i < 10; i++) yield return new WaitForFixedUpdate();
            Destroy(joint); joint = null;
        }
        float t = 0f;
        while (t < settleSeconds) { yield return new WaitForFixedUpdate(); t += Time.fixedDeltaTime; if (stopStep < 0) CheckContacts(armHits, toolHits, ref appleShelf, ref firstContact); }

        var bb = basketCols[0].bounds; foreach (var bc in basketCols) bb.Encapsulate(bc.bounds); bb.Expand(new Vector3(0f, 0.06f, 0f));
        bool placed = joint == null && bb.Contains(apple.position) && appleRb.linearVelocity.magnitude < 0.02f;
        var ap = apple.position; var bcen = basket.position;
        float goalDist = new Vector2(ap.x - bcen.x, ap.z - bcen.z).magnitude;
        bool coll = stopStep >= 0;

        string fdir = Path.Combine(cdir, "final"); Directory.CreateDirectory(fdir);
        int maskPx = WriteAppleMask(Path.Combine(fdir, "target_mask.png"));
        var q4 = apple.rotation;
        File.WriteAllText(Path.Combine(fdir, "final_state.json"),
            "{\n" +
            $"  \"labels\": {{\"goal_reached\": {B(placed)}, \"secondary_collision\": {B(coll)}, \"target_fell\": false, \"goal_distance_m\": {N(goalDist)}, " +
            $"\"protective_stop\": {B(coll)}, \"stop_step\": {stopStep}, \"arm_shelf\": {B(armHits.Count > 0)}, \"tool_contact\": {B(toolHits.Count > 0)}, \"apple_shelf\": {B(appleShelf)}}},\n" +
            $"  \"contacts\": {{\"arm\": \"{string.Join("|", armHits)}\", \"tool\": \"{string.Join("|", toolHits)}\", \"first_point_world\": [{N(firstContact.x)}, {N(firstContact.y)}, {N(firstContact.z)}]}},\n" +
            $"  \"objects\": [{{\"name\": \"Target\", \"what\": \"apple\", \"final_position_world\": [{N(ap.x)}, {N(ap.y)}, {N(ap.z)}], \"final_rotation_world_xyzw\": [{N(q4.x)}, {N(q4.y)}, {N(q4.z)}, {N(q4.w)}]}}],\n" +
            $"  \"target_mask_pixels\": {maskPx},\n" +
            $"  \"max_tcp_err_m\": {N(maxTcpErr)}\n}}\n", new UTF8Encoding(false));

        if (joint != null) Destroy(joint);
        yield return new WaitForFixedUpdate();
        appleRb.isKinematic = true;
        apple.position = new Vector3(0, -5, 0);
    }

    bool CheckContacts(SortedSet<string> arm, SortedSet<string> tool, ref bool appleShelf, ref Vector3 first)
    {
        bool any = false;
        foreach (var (link, board, sep, point) in ShelfContactProbe.Hits)
        {
            bool isBasket = basketCols.Any(c => c.name == board);
            bool isTool = toolLinks.Contains(link);
            (isTool ? tool : arm).Add(link + ">" + board);
            if (float.IsNaN(first.x)) first = point;
            any = true;
            _ = isBasket;
        }
        ShelfContactProbe.Hits.Clear();
        foreach (var (c, ln, isTool) in robotCols)
            foreach (var s in shelfCols.Concat(basketCols))
            {
                if (!c.bounds.Intersects(s.bounds)) continue;
                if (Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, s, s.transform.position, s.transform.rotation, out _, out _))
                {
                    any = true; (isTool ? tool : arm).Add(ln + ">" + s.name);
                    if (float.IsNaN(first.x)) first = c.ClosestPoint(s.bounds.center);
                }
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

    // ---------- 画像 ----------

    void SetRobotVisible(bool v) { foreach (var r in robotRenderers) if (r != null) r.enabled = v; }

    void CaptureRgb(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        cam.Render();
        var prev = RenderTexture.active; RenderTexture.active = rt;
        var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0); tex.Apply();
        RenderTexture.active = prev;
        File.WriteAllBytes(path, tex.EncodeToPNG());
        Destroy(tex);
    }

    // 見える面のメッシュ（ロボット・りんご以外）を CPU で描いて depth（カメラの前方向の距離、m）を作る
    List<(Vector3[] v, int[] t, bool apple)> SceneMeshes(bool includeApple)
    {
        var list = new List<(Vector3[], int[], bool)>();
        foreach (var mf in FindObjectsOfType<MeshFilter>())
        {
            var r = mf.GetComponent<Renderer>();
            if (r == null || !r.enabled || !mf.gameObject.activeInHierarchy || mf.sharedMesh == null || !mf.sharedMesh.isReadable) continue;
            if (mf.transform.IsChildOf(rec.robotRoot)) continue;
            bool isApple = mf.transform.IsChildOf(apple);
            if (isApple && !includeApple) continue;
            var m = mf.sharedMesh; var src = m.vertices; var v = new Vector3[src.Length];
            for (int k = 0; k < src.Length; k++) v[k] = mf.transform.TransformPoint(src[k]);
            list.Add((v, m.triangles, isApple));
        }
        return list;
    }

    float[] Raster(List<(Vector3[] v, int[] t, bool apple)> meshes, out bool[] isApple)
    {
        int W = cam.pixelWidth, H = cam.pixelHeight;
        var z = new float[W * H]; for (int i = 0; i < z.Length; i++) z[i] = float.PositiveInfinity;
        isApple = new bool[W * H];
        Matrix4x4 V = cam.worldToCameraMatrix, PV = cam.projectionMatrix * V;
        float near = cam.nearClipPlane;
        foreach (var (v, t, ap) in meshes)
        {
            int n = v.Length; var sx = new float[n]; var sy = new float[n]; var sz = new float[n];
            for (int k = 0; k < n; k++)
            {
                var c = V.MultiplyPoint3x4(v[k]); sz[k] = -c.z;
                var clip = PV * new Vector4(v[k].x, v[k].y, v[k].z, 1f); float iw = 1f / clip.w;
                sx[k] = (clip.x * iw + 1f) * 0.5f * W - 0.5f; sy[k] = H - (clip.y * iw + 1f) * 0.5f * H - 0.5f;
            }
            for (int k = 0; k < t.Length; k += 3)
            {
                int a = t[k], b = t[k + 1], c = t[k + 2];
                if (sz[a] <= near || sz[b] <= near || sz[c] <= near) continue;
                Tri(sx[a], sy[a], sz[a], sx[b], sy[b], sz[b], sx[c], sy[c], sz[c], W, H, z, isApple, ap);
            }
        }
        for (int i = 0; i < z.Length; i++) if (float.IsPositiveInfinity(z[i])) z[i] = 0f;
        return z;
    }

    static void Tri(float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, int W, int H, float[] zbuf, bool[] id, bool tag)
    {
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (Mathf.Abs(area) < 1e-12f) return;
        int xmin = Mathf.Max(0, Mathf.CeilToInt(Mathf.Min(x0, Mathf.Min(x1, x2)))), xmax = Mathf.Min(W - 1, Mathf.FloorToInt(Mathf.Max(x0, Mathf.Max(x1, x2))));
        int ymin = Mathf.Max(0, Mathf.CeilToInt(Mathf.Min(y0, Mathf.Min(y1, y2)))), ymax = Mathf.Min(H - 1, Mathf.FloorToInt(Mathf.Max(y0, Mathf.Max(y1, y2))));
        if (xmin > xmax || ymin > ymax) return;
        float inv = 1f / area, iz0 = 1f / z0, iz1 = 1f / z1, iz2 = 1f / z2;
        for (int y = ymin; y <= ymax; y++)
            for (int x = xmin; x <= xmax; x++)
            {
                float w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) * inv, w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) * inv, w2 = 1f - w0 - w1;
                if (w0 < 0f || w1 < 0f || w2 < 0f) continue;
                float z = 1f / (w0 * iz0 + w1 * iz1 + w2 * iz2); int idx = y * W + x;
                if (z < zbuf[idx]) { zbuf[idx] = z; id[idx] = tag; }
            }
    }

    void WriteSceneDepth(string path)
    {
        var z = Raster(SceneMeshes(false), out _);
        int W = cam.pixelWidth, H = cam.pixelHeight;
        var tex = new Texture2D(W, H, TextureFormat.RGBAFloat, false, true);
        var px = new Color[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { float d = z[y * W + x]; px[(H - 1 - y) * W + x] = new Color(d, d, d, 1f); }
        tex.SetPixels(px); tex.Apply();
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, tex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat));
        Destroy(tex);
    }

    int WriteAppleMask(string path)
    {
        Physics.SyncTransforms();
        Raster(SceneMeshes(true), out var isApple);
        int W = cam.pixelWidth, H = cam.pixelHeight, n = 0;
        var mask = new byte[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (isApple[y * W + x]) { mask[(H - 1 - y) * W + x] = 255; n++; }
        File.WriteAllBytes(path, ImageConversion.EncodeArrayToPNG(mask, GraphicsFormat.R8_UNorm, (uint)W, (uint)H));
        return n;
    }

    // ---------- 書き出し ----------

    string PlanJson(PlannedTrajectory tr)
    {
        var sb = new StringBuilder("{\n  \"schema\": \"task2_planned_v1\",\n");
        sb.Append($"  \"dt\": {N(tr.dt)},\n  \"num_steps\": {tr.q.Count},\n  \"joint_names\": [{string.Join(", ", kin.JointNames.Select(n => "\"" + n + "\""))}],\n  \"steps\": [\n");
        for (int i = 0; i < tr.q.Count; i++)
        {
            sb.Append($"    {{\"t\": {N(tr.t[i])}, \"phase\": \"{tr.phase[i]}\", \"q\": [{string.Join(", ", tr.q[i].Select(v => N(v)))}], \"tcp\": [{N(tr.tcp[i].x)}, {N(tr.tcp[i].y)}, {N(tr.tcp[i].z)}]}}");
            sb.Append(i < tr.q.Count - 1 ? ",\n" : "\n");
        }
        sb.Append("  ]\n}\n");
        return sb.ToString();
    }

    static string N(float v) => float.IsNaN(v) ? "null" : v.ToString("G7", IC);
    static string B(bool b) => b ? "true" : "false";

    void OnDestroy() { if (cam != null) cam.targetTexture = null; if (rt != null) rt.Release(); Time.timeScale = 1f; }
}
