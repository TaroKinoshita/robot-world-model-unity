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
/// 3 つ目のタスク（前腕で物を押す）のデータを作る。Task2Pilot シーンで、Task2Rec を無効にしてこれを有効にして Play する。
/// 1 シーン = 机の上の背の高い箱 1 つの配置。手先は箱の横を、区画のときと同じ下向き・直線で動く（手先は箱に触れない配置だけ使う）。
/// 候補 = 手先の入り方（dx）× 肩の左右の解（A / B）。A と B は同じ手先の経路（matched pair）。腕は止めない（押したら箱が動く）。
/// 書き出しの形は Task2Recorder と同じ（Learning/wm_cache_t3.py が読む）。初期画像には箱が写る（ロボットだけ除く）。
/// </summary>
public class Task3Recorder : MonoBehaviour
{
    [Header("出力")]
    public string outputRoot = "Episodes/_task3_pilot";
    public int startSeed = 20000;
    public int sceneCount = 20;
    public float timeScale = 8f;
    public bool autoRun = true;

    [Header("配置（一様乱数の範囲）")]
    public Vector2 tcpXRange = new Vector2(-0.07f, 0.13f);
    public Vector2 tcpZRange = new Vector2(0.53f, 0.60f);    // 経路の終点（ワールド z）
    public float tcpAboveTable = 0.06f;
    public Vector2 boxLatRange = new Vector2(0.10f, 0.20f);  // 経路からの横のずれ（左右はランダム）
    public Vector2 boxAlongRange = new Vector2(-0.03f, 0.12f); // 経路の始点からの奥行き
    public Vector2 boxWRange = new Vector2(0.06f, 0.10f), boxDRange = new Vector2(0.05f, 0.08f), boxHRange = new Vector2(0.28f, 0.42f);
    public float boxMass = 0.4f;
    public float gripMargin = 0.10f;

    [Header("経路")]
    public float tableTop = 0.773f, approachDist = 0.30f, speed = 0.10f;
    public float[] approachDx = { -0.05f, 0f, 0.05f };
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
    Transform box;
    Rigidbody boxRb;
    BoxCollider boxCol;
    List<(Collider col, string link, bool tool)> robotCols;
    HashSet<string> toolLinks;
    Renderer[] robotRenderers;
    string root;
    static readonly CultureInfo IC = CultureInfo.InvariantCulture;

    struct Layout { public float tcpX, tcpZ, lat, along, w, d, h; public Vector3 boxPos; }

    void Start() { if (autoRun) StartCoroutine(Run()); }

    IEnumerator Run()
    {
        for (int i = 0; i < 5; i++) yield return null;
        rec = FindObjectOfType<EpisodeRecorder>(true);
        kin = ArmKinematics.Build(rec.robotRoot, rec.toolLinkName, rec.armJointNames, rec.fingerPadA, rec.fingerPadB, out string err);
        if (kin == null) { Debug.LogError("[T3R] kinematics: " + err); yield break; }
        rr = RobotRasterizer.Build(kin, rec.robotRoot);
        cam = Camera.main;
        rt = new RenderTexture(256, 256, 24, RenderTextureFormat.ARGB32) { name = "T3R_RT" };
        cam.targetTexture = rt;
        // 棚とりんごは使わない
        var shelfGo = GameObject.Find("Task2Shelf/Shelf"); var matSrc = shelfGo.transform.Find("Bottom").GetComponent<Renderer>().sharedMaterial;
        shelfGo.SetActive(false);
        var appleGo = GameObject.Find("Task2Shelf/Apple"); if (appleGo != null) appleGo.SetActive(false);
        // 机を奥へ延ばす（z −0.15〜0.65 → −0.15〜1.05）。背の高い箱が前に倒れると、元の机では奥の縁から落ちたため
        var table = FindObjectsOfType<BoxCollider>().First(c => c.name == "Table").transform;
        table.position = new Vector3(table.position.x, table.position.y, 0.45f);
        table.localScale = new Vector3(table.localScale.x, table.localScale.y, 1.20f);
        // 押す箱
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube); go.name = "PushBox";
        box = go.transform; boxCol = go.GetComponent<BoxCollider>();
        var mat = new Material(matSrc); mat.SetColor("_BaseColor", new Color(0.72f, 0.55f, 0.36f)); go.GetComponent<Renderer>().sharedMaterial = mat;
        boxCol.material = new PhysicsMaterial("box") { dynamicFriction = 0.5f, staticFriction = 0.6f, bounciness = 0f };
        boxRb = go.AddComponent<Rigidbody>(); boxRb.mass = boxMass; boxRb.interpolation = RigidbodyInterpolation.None;
        boxRb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic; boxRb.isKinematic = true;
        boxRb.maxDepenetrationVelocity = 0.5f; boxRb.solverIterations = 20;
        go.AddComponent<ShelfContactProbe>();

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
        }
        toolLinks = new HashSet<string>(robotCols.Where(r => r.tool).Select(r => r.link));
        robotRenderers = rec.robotRoot.GetComponentsInChildren<Renderer>(true).Where(r => r.enabled).ToArray();
        foreach (var ab in rec.robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (Array.IndexOf(kin.JointBodies, ab) >= 0) { var d = ab.xDrive; d.stiffness = stiffness; d.damping = damping; ab.xDrive = d; }
        ShelfContactProbe.Enabled = true;

        root = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outputRoot));
        Directory.CreateDirectory(root);
        Time.timeScale = timeScale;
        var t0 = DateTime.Now; int done = 0, skipped = 0;
        Debug.Log($"[T3R] start: {root}, seeds {startSeed}..{startSeed + sceneCount - 1}, timeScale {timeScale}");
        for (int i = 0; i < sceneCount; i++)
        {
            int seed = startSeed + i;
            string sid = $"scene_{seed:D5}";
            string sd = Path.Combine(root, sid);
            if (File.Exists(Path.Combine(sd, "metadata.json"))) { skipped++; continue; }
            if (Directory.Exists(sd)) Directory.Delete(sd, true);
            yield return RunScene(seed, sid, sd);
            done++;
            Debug.Log($"[T3R] {sid} done ({done} new, {skipped} skipped, {(DateTime.Now - t0).TotalMinutes:F1} min)");
        }
        Time.timeScale = 1f;
        cam.targetTexture = null;
        File.WriteAllText(Path.Combine(root, "_batch_done.txt"), $"{DateTime.Now:s} new {done} skipped {skipped}\n");
        Debug.Log($"[T3R] DONE new {done}, skipped {skipped}");
    }

    // ---------- 配置 ----------

    Quaternion DownRot()
    {
        var rip = FindObjectOfType<RobotInitialPose>();
        kin.TcpPose(rip.jointDegrees.Select(d => d * Mathf.Deg2Rad).ToArray(), out _, out var rot);
        return rot;
    }

    // 手先（グリッパー）が箱に近づかない配置になるまで引き直す。始点で腕が箱に重なる配置も除く
    Layout Sample(System.Random r, Quaternion rot, out List<(int mp, float dx, PlannedTrajectory a, PlannedTrajectory b)> pairs)
    {
        float U(Vector2 v) => (float)(v.x + r.NextDouble() * (v.y - v.x));
        for (int attempt = 0; ; attempt++)
        {
            var L = new Layout { tcpX = U(tcpXRange), tcpZ = U(tcpZRange), lat = U(boxLatRange) * (r.NextDouble() < 0.5 ? -1 : 1), along = U(boxAlongRange), w = U(boxWRange), d = U(boxDRange), h = U(boxHRange) };
            L.boxPos = new Vector3(L.tcpX + L.lat, tableTop + L.h / 2, L.tcpZ - approachDist + L.along);
            var bb = new Bounds(L.boxPos, new Vector3(L.w, L.h, L.d));
            var P = new Vector3(L.tcpX, tableTop + tcpAboveTable, L.tcpZ);
            pairs = new List<(int, float, PlannedTrajectory, PlannedTrajectory)>();
            bool bad = false;
            foreach (float dx in approachDx)
            {
                var front = P + new Vector3(dx, 0, -approachDist);
                var trA = Plan(P, front, rot, seedADeg); var trB = Plan(P, front, rot, seedBDeg);
                if (trA == null || trB == null) continue;
                if (GripperNear(trA, bb) || ArmNear(trA.q[0], bb) || ArmNear(trB.q[0], bb)) { bad = true; break; }
                pairs.Add((pairs.Count, dx, trA, trB));
            }
            if (!bad && pairs.Count > 0) return L;
            if (attempt > 500) { Debug.LogError("[T3R] 配置が見つからない"); return L; }
        }
    }

    bool GripperNear(PlannedTrajectory tr, Bounds bb)
    {
        int n = kin.LinkNames.Length; var pos = new Vector3[n]; var rot = new Quaternion[n];
        foreach (var q in tr.q)
        {
            kin.LinkPoses(q, pos, rot); kin.TcpPose(q, out var tcp, out _);
            for (int m = 0; m <= 8; m++) if (bb.SqrDistance(Vector3.Lerp(pos[n - 1], tcp, m / 8f)) < gripMargin * gripMargin) return true;
        }
        return false;
    }

    bool ArmNear(float[] q, Bounds bb)
    {
        int n = kin.LinkNames.Length; var pos = new Vector3[n]; var rot = new Quaternion[n];
        kin.LinkPoses(q, pos, rot);
        for (int j = 2; j < n; j++) for (int m = 0; m <= 8; m++) if (bb.SqrDistance(Vector3.Lerp(pos[j - 1], pos[j], m / 8f)) < 0.07f * 0.07f) return true;
        return false;
    }

    // ---------- 1 シーン ----------

    IEnumerator RunScene(int seed, string sid, string sd)
    {
        var rng = new System.Random(seed);
        var rot = DownRot();
        var L = Sample(rng, rot, out var pairs);
        box.localScale = new Vector3(L.w, L.h, L.d);
        ResetBox(L);
        Directory.CreateDirectory(sd);

        SetRobotVisible(false);
        for (int i = 0; i < 4; i++) yield return null;   // 箱を動かした直後は TAA・モーションブラーで残像が出るので、数フレーム待つ
        yield return new WaitForEndOfFrame();
        CaptureRgb(Path.Combine(sd, "initial", "robot_free", "rgb.png"));
        WriteSceneDepth(Path.Combine(sd, "initial", "robot_free", "depth.exr"));
        SetRobotVisible(true);
        var exporter = cam.GetComponent<CameraExporter>();
        if (exporter == null) exporter = cam.gameObject.AddComponent<CameraExporter>();
        exporter.targetCamera = cam;
        exporter.ExportToFile(Path.Combine(sd, "camera.json"));

        var cands = new List<(string cid, int mp, string branch, float dx, PlannedTrajectory tr)>();
        foreach (var p in pairs)
        {
            cands.Add(($"c{cands.Count:D3}", p.mp, "shoulder_a", p.dx, p.a));
            cands.Add(($"c{cands.Count:D3}", p.mp, "shoulder_b", p.dx, p.b));
        }
        var cj = new StringBuilder("{\n  \"scene_id\": \"" + sid + "\",\n  \"candidates\": [\n");
        for (int k = 0; k < cands.Count; k++)
        {
            var c = cands[k];
            string cdir = Path.Combine(sd, "candidates", c.cid);
            Directory.CreateDirectory(cdir);
            ResetBox(L);
            ActionRepresentations.Write(kin, c.tr, cam, cdir, raster, sid, c.cid, rr);
            File.WriteAllText(Path.Combine(cdir, "planned_trajectory.json"), PlanJson(c.tr), new UTF8Encoding(false));
            yield return Execute(c.tr, cdir, L);
            cj.Append($"    {{\"candidate_id\": \"{c.cid}\", \"matched_pair_id\": \"mp{c.mp}\", \"branch\": \"{c.branch}\", \"approach_dx\": {N(c.dx)}, \"note\": \"matched pair {c.mp}: {c.branch}, same TCP path\"}}");
            cj.Append(k < cands.Count - 1 ? ",\n" : "\n");
        }
        cj.Append("  ]\n}\n");
        File.WriteAllText(Path.Combine(sd, "candidates.json"), cj.ToString(), new UTF8Encoding(false));

        var bp = L.boxPos; var P = new Vector3(L.tcpX, tableTop + tcpAboveTable, L.tcpZ);
        File.WriteAllText(Path.Combine(sd, "scene_generation.json"),
            "{\n" +
            $"  \"seed\": {seed},\n" +
            $"  \"layout\": {{\"tcp_x\": {N(L.tcpX)}, \"tcp_z\": {N(L.tcpZ)}, \"box_lateral\": {N(L.lat)}, \"box_along\": {N(L.along)}, \"box_w\": {N(L.w)}, \"box_d\": {N(L.d)}, \"box_h\": {N(L.h)}}},\n" +
            $"  \"goal\": {{\"center_world\": [{N(bp.x)}, {N(tableTop)}, {N(bp.z)}], \"what\": \"box initial footprint center\"}},\n" +
            $"  \"target\": {{\"position_world\": [{N(bp.x)}, {N(bp.y)}, {N(bp.z)}], \"yaw_deg\": 0, \"what\": \"box center (initial)\"}},\n" +
            $"  \"secondary\": {{\"position_world\": [{N(P.x)}, {N(P.y)}, {N(P.z)}], \"height_m\": {N(L.h)}, \"what\": \"TCP path end\"}},\n" +
            $"  \"candidates\": {cands.Count}\n}}\n", new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(sd, "scene_initial.json"),
            $"{{\n  \"seed\": {seed},\n  \"task\": {{\"task_variant\": \"T3\", \"pair_id\": \"{sid.Replace("scene_", "pair_")}\"}},\n  \"unity_version\": \"{Application.unityVersion}\"\n}}\n", new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(sd, "metadata.json"),
            $"{{\n  \"schema\": \"task3_scene_v1\",\n  \"scene_id\": \"{sid}\",\n  \"created\": \"{DateTime.Now:s}\",\n  \"fixed_dt\": {N(Time.fixedDeltaTime)},\n  \"time_scale\": {N(timeScale)},\n  \"candidates\": {cands.Count}\n}}\n", new UTF8Encoding(false));
    }

    void ResetBox(Layout L)
    {
        boxRb.isKinematic = true;
        box.SetPositionAndRotation(L.boxPos, Quaternion.identity);
        Physics.SyncTransforms();
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
        for (int k = 0; k < 10; k++) Add((float[])q.Clone(), "hold");
        return tr;
    }

    // ---------- 実行（止めない。押したら箱が動く） ----------

    IEnumerator Execute(PlannedTrajectory tr, string cdir, Layout L)
    {
        for (int j = 0; j < kin.Dof; j++)
        {
            var ab = kin.JointBodies[j];
            ab.jointPosition = new ArticulationReducedSpace(tr.q[0][j]);
            ab.jointVelocity = new ArticulationReducedSpace(0f);
            var d = ab.xDrive; d.target = tr.q[0][j] * Mathf.Rad2Deg; d.targetVelocity = 0; ab.xDrive = d;
        }
        for (int i = 0; i < 10; i++) yield return new WaitForFixedUpdate();
        ResetBox(L);
        boxRb.isKinematic = false; boxRb.linearVelocity = Vector3.zero; boxRb.angularVelocity = Vector3.zero;
        for (int i = 0; i < 5; i++) yield return new WaitForFixedUpdate();
        ShelfContactProbe.Hits.Clear();

        Vector3 firstContact = new Vector3(float.NaN, float.NaN, float.NaN);
        var armHits = new SortedSet<string>(); var toolHits = new SortedSet<string>();
        float maxTcpErr = 0f, minArm = 9f, minTool = 9f;
        for (int s = 1; s < tr.q.Count; s++)
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
            CheckContacts(armHits, toolHits, ref firstContact);
            foreach (var (c, ln, isTool) in robotCols)
            {
                var pb = boxCol.ClosestPoint(c.bounds.center); float dd = Vector3.Distance(c.ClosestPoint(pb), pb);
                if (isTool) minTool = Mathf.Min(minTool, dd); else minArm = Mathf.Min(minArm, dd);
            }
        }
        float t = 0f;
        while (t < settleSeconds) { yield return new WaitForFixedUpdate(); t += Time.fixedDeltaTime; CheckContacts(armHits, toolHits, ref firstContact); }

        var bp = box.position; var q4 = box.rotation;
        float disp = new Vector2(bp.x - L.boxPos.x, bp.z - L.boxPos.z).magnitude;
        bool tipped = Vector3.Angle(box.up, Vector3.up) > 30f;
        bool armContact = armHits.Count > 0;

        string fdir = Path.Combine(cdir, "final"); Directory.CreateDirectory(fdir);
        SetRobotVisible(false);
        int maskPx = WriteBoxMask(Path.Combine(fdir, "target_mask.png"));
        SetRobotVisible(true);
        File.WriteAllText(Path.Combine(fdir, "final_state.json"),
            "{\n" +
            $"  \"labels\": {{\"goal_reached\": {B(tipped)}, \"secondary_collision\": {B(armContact)}, \"target_fell\": {B(tipped)}, \"goal_distance_m\": {N(disp)}, " +
            $"\"arm_contact\": {B(armContact)}, \"tool_contact\": {B(toolHits.Count > 0)}, \"tipped\": {B(tipped)}, \"displacement_m\": {N(disp)}}},\n" +
            $"  \"contacts\": {{\"arm\": \"{string.Join("|", armHits)}\", \"tool\": \"{string.Join("|", toolHits)}\", \"first_point_world\": [{N(firstContact.x)}, {N(firstContact.y)}, {N(firstContact.z)}]}},\n" +
            $"  \"objects\": [{{\"name\": \"Target\", \"what\": \"box\", \"final_position_world\": [{N(bp.x)}, {N(bp.y)}, {N(bp.z)}], \"final_rotation_world_xyzw\": [{N(q4.x)}, {N(q4.y)}, {N(q4.z)}, {N(q4.w)}]}}],\n" +
            $"  \"target_mask_pixels\": {maskPx},\n" +
            $"  \"min_arm_dist_m\": {N(minArm)},\n  \"min_tool_dist_m\": {N(minTool)},\n" +
            $"  \"max_tcp_err_m\": {N(maxTcpErr)}\n}}\n", new UTF8Encoding(false));
        ResetBox(L);
    }

    void CheckContacts(SortedSet<string> arm, SortedSet<string> tool, ref Vector3 first)
    {
        foreach (var (link, _, _, point) in ShelfContactProbe.Hits)
        {
            (toolLinks.Contains(link) ? tool : arm).Add(link);
            if (float.IsNaN(first.x)) first = point;
        }
        ShelfContactProbe.Hits.Clear();
        foreach (var (c, ln, isTool) in robotCols)
        {
            if (!c.bounds.Intersects(boxCol.bounds)) continue;
            if (Physics.ComputePenetration(c, c.transform.position, c.transform.rotation, boxCol, box.position, box.rotation, out _, out _))
            {
                (isTool ? tool : arm).Add(ln);
                if (float.IsNaN(first.x)) first = c.ClosestPoint(box.position);
            }
        }
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

    List<(Vector3[] v, int[] t, bool tgt)> SceneMeshes()
    {
        var list = new List<(Vector3[], int[], bool)>();
        foreach (var mf in FindObjectsOfType<MeshFilter>())
        {
            var r = mf.GetComponent<Renderer>();
            if (r == null || !r.enabled || !mf.gameObject.activeInHierarchy || mf.sharedMesh == null || !mf.sharedMesh.isReadable) continue;
            if (mf.transform.IsChildOf(rec.robotRoot)) continue;
            var m = mf.sharedMesh; var src = m.vertices; var v = new Vector3[src.Length];
            for (int k = 0; k < src.Length; k++) v[k] = mf.transform.TransformPoint(src[k]);
            list.Add((v, m.triangles, mf.transform == box));
        }
        return list;
    }

    float[] Raster(List<(Vector3[] v, int[] t, bool tgt)> meshes, out bool[] isTgt)
    {
        int W = cam.pixelWidth, H = cam.pixelHeight;
        var z = new float[W * H]; for (int i = 0; i < z.Length; i++) z[i] = float.PositiveInfinity;
        isTgt = new bool[W * H];
        Matrix4x4 V = cam.worldToCameraMatrix, PV = cam.projectionMatrix * V;
        float near = cam.nearClipPlane;
        foreach (var (v, t, tg) in meshes)
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
                Tri(sx[a], sy[a], sz[a], sx[b], sy[b], sz[b], sx[c], sy[c], sz[c], W, H, z, isTgt, tg);
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
        Physics.SyncTransforms();
        var z = Raster(SceneMeshes(), out _);
        int W = cam.pixelWidth, H = cam.pixelHeight;
        var tex = new Texture2D(W, H, TextureFormat.RGBAFloat, false, true);
        var px = new Color[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { float d = z[y * W + x]; px[(H - 1 - y) * W + x] = new Color(d, d, d, 1f); }
        tex.SetPixels(px); tex.Apply();
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, tex.EncodeToEXR(Texture2D.EXRFlags.OutputAsFloat));
        Destroy(tex);
    }

    int WriteBoxMask(string path)
    {
        Physics.SyncTransforms();
        Raster(SceneMeshes(), out var isTgt);
        int W = cam.pixelWidth, H = cam.pixelHeight, n = 0;
        var mask = new byte[W * H];
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) if (isTgt[y * W + x]) { mask[(H - 1 - y) * W + x] = 255; n++; }
        File.WriteAllBytes(path, ImageConversion.EncodeArrayToPNG(mask, GraphicsFormat.R8_UNorm, (uint)W, (uint)H));
        return n;
    }

    string PlanJson(PlannedTrajectory tr)
    {
        var sb = new StringBuilder("{\n  \"schema\": \"task3_planned_v1\",\n");
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
