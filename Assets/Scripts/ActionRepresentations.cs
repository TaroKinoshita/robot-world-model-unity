using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

/// <summary>raster 表現の描き方</summary>
[System.Serializable]
public class ActionRasterSettings
{
    [Tooltip("何ステップごとに画像を作るか(dt 0.02 s × 5 = 10 Hz)。最後のステップは必ず作る")]
    public int rasterStride = 5;
    public float eefRadiusPx = 3f;          // EEF の点の半径
    public float bodyLineRadiusPx = 1.5f;   // 腕の線の太さ(半径)
    public float bodyJointRadiusPx = 2.5f;  // 関節の点の半径
}

/// <summary>
/// TODO5-4: Planned trajectory から 4 種類の action representation を作る。
///   EEF numeric      : TCP の位置 + tool の回転 + グリッパー状態(毎ステップ)
///   Full-body numeric: 関節角 + 腕の各リンク原点と TCP の 3D 位置(毎ステップ)
///   EEF raster       : TCP を画像上の点として描いた 8bit 画像
///   Full-body raster : 腕の骨格(リンク原点を結ぶ線)を描いた 8bit 画像
/// 画像は camera.json と同じカメラ・同じ規約(OpenCV、左上ピクセルの中心が (0,0))で投影する。
/// </summary>
public static class ActionRepresentations
{
    public static void Write(ArmKinematics kin, PlannedTrajectory tr, Camera cam, string candDir,
                             ActionRasterSettings s, string sceneId, string cid)
    {
        string dir = Path.Combine(candDir, "actions");
        if (Directory.Exists(dir)) Directory.Delete(dir, true);   // 前の実行の画像を消す
        string eefDir = Path.Combine(dir, "eef_raster");
        string bodyDir = Path.Combine(dir, "fullbody_raster");
        Directory.CreateDirectory(eefDir);
        Directory.CreateDirectory(bodyDir);

        int W = cam.pixelWidth, H = cam.pixelHeight;
        int K = kin.KeypointNames.Length;
        var pts = new Vector3[K];
        var uv = new Vector2[K];
        var vis = new bool[K];
        var eefImg = new byte[W * H];
        var bodyImg = new byte[W * H];
        int stride = Mathf.Max(1, s.rasterStride);

        var eef = new StringBuilder("step,t,phase,tcp_x,tcp_y,tcp_z,rot_x,rot_y,rot_z,rot_w,gripper_open\n");
        var body = new StringBuilder("step,t,phase," +
            string.Join(",", kin.JointNames.Select(n => "q_" + n)) + "," +
            string.Join(",", kin.KeypointNames.SelectMany(n => new[] { n + "_x", n + "_y", n + "_z" })) + "\n");
        var rasterSteps = new List<int>();

        for (int i = 0; i < tr.q.Count; i++)
        {
            var q = tr.q[i];
            kin.Keypoints(q, pts);
            Vector3 c = tr.tcp[i];
            Quaternion r = tr.toolRot[i];

            eef.Append(i).Append(',').Append(N(tr.t[i])).Append(',').Append(tr.phase[i])
               .Append(',').Append(N(c.x)).Append(',').Append(N(c.y)).Append(',').Append(N(c.z))
               .Append(',').Append(N(r.x)).Append(',').Append(N(r.y)).Append(',').Append(N(r.z)).Append(',').Append(N(r.w))
               .Append(",1\n");

            body.Append(i).Append(',').Append(N(tr.t[i])).Append(',').Append(tr.phase[i]);
            foreach (var v in q) body.Append(',').Append(N(v));
            foreach (var p in pts) body.Append(',').Append(N(p.x)).Append(',').Append(N(p.y)).Append(',').Append(N(p.z));
            body.Append('\n');

            if (i % stride == 0 || i == tr.q.Count - 1)
            {
                for (int k = 0; k < K; k++) vis[k] = Project(cam, pts[k], H, out uv[k]);
                System.Array.Clear(eefImg, 0, eefImg.Length);
                System.Array.Clear(bodyImg, 0, bodyImg.Length);

                if (vis[K - 1]) Disk(eefImg, W, H, uv[K - 1], s.eefRadiusPx);
                for (int k = 0; k < K - 1; k++)
                    if (vis[k] && vis[k + 1]) Line(bodyImg, W, H, uv[k], uv[k + 1], s.bodyLineRadiusPx);
                for (int k = 0; k < K; k++)
                    if (vis[k]) Disk(bodyImg, W, H, uv[k], s.bodyJointRadiusPx);

                string name = $"step_{i:D4}.png";
                File.WriteAllBytes(Path.Combine(eefDir, name), Encode(eefImg, W, H));
                File.WriteAllBytes(Path.Combine(bodyDir, name), Encode(bodyImg, W, H));
                rasterSteps.Add(i);
            }
        }

        var utf8 = new UTF8Encoding(false);
        File.WriteAllText(Path.Combine(dir, "eef_numeric.csv"), eef.ToString(), utf8);
        File.WriteAllText(Path.Combine(dir, "fullbody_numeric.csv"), body.ToString(), utf8);

        string meta = "{\n" +
            "  \"schema\": \"action_representations_v1\",\n" +
            $"  \"scene_id\": {Q(sceneId)},\n" +
            $"  \"candidate_id\": {Q(cid)},\n" +
            "  \"source\": \"planned_trajectory.json (planned, not executed)\",\n" +
            $"  \"dt\": {N(tr.dt)},\n" +
            $"  \"num_steps\": {tr.q.Count},\n" +
            "  \"frames\": {\"world\": \"Unity world: left-handed, Y-up, meters\", \"image\": \"same camera as ../../camera.json; OpenCV pixel convention\"},\n" +
            "  \"eef_numeric\": {\"file\": \"eef_numeric.csv\", \"point\": \"TCP (midpoint of finger pads)\", \"rotation\": \"tool link rotation, quaternion xyzw\", \"gripper_open\": \"1 = open (constant)\"},\n" +
            "  \"fullbody_numeric\": {\"file\": \"fullbody_numeric.csv\", \"joints_rad\": [" + string.Join(", ", kin.JointNames.Select(Q)) + "], " +
            "\"keypoints\": [" + string.Join(", ", kin.KeypointNames.Select(Q)) + "], \"keypoint_definition\": \"link origins along the chain root -> tool, then TCP\"},\n" +
            $"  \"eef_raster\": {{\"dir\": \"eef_raster\", \"width\": {W}, \"height\": {H}, \"encoding\": \"8-bit gray PNG, 255 = TCP disk, 0 = background\", \"radius_px\": {N(s.eefRadiusPx)}}},\n" +
            $"  \"fullbody_raster\": {{\"dir\": \"fullbody_raster\", \"width\": {W}, \"height\": {H}, \"encoding\": \"8-bit gray PNG, 255 = skeleton (lines between consecutive keypoints + keypoint disks), 0 = background, no occlusion handling\", " +
            $"\"line_radius_px\": {N(s.bodyLineRadiusPx)}, \"joint_radius_px\": {N(s.bodyJointRadiusPx)}}},\n" +
            $"  \"raster_stride\": {stride},\n" +
            "  \"raster_steps\": [" + string.Join(", ", rasterSteps) + "]\n}\n";
        File.WriteAllText(Path.Combine(dir, "actions_meta.json"), meta, utf8);
    }

    // world → OpenCV 規約のピクセル座標。カメラの前にあれば true
    static bool Project(Camera cam, Vector3 w, int H, out Vector2 uv)
    {
        Vector3 sp = cam.WorldToScreenPoint(w);                // 左下原点、ピクセルの角が整数
        uv = new Vector2(sp.x - 0.5f, (H - sp.y) - 0.5f);
        return sp.z > 0f;
    }

    // img は OpenCV の向き(行 0 = 画像の上)で持つ
    static void Disk(byte[] img, int W, int H, Vector2 c, float r)
    {
        int x0 = Mathf.FloorToInt(c.x - r), x1 = Mathf.CeilToInt(c.x + r);
        int y0 = Mathf.FloorToInt(c.y - r), y1 = Mathf.CeilToInt(c.y + r);
        float r2 = r * r;
        for (int y = Mathf.Max(0, y0); y <= Mathf.Min(H - 1, y1); y++)
            for (int x = Mathf.Max(0, x0); x <= Mathf.Min(W - 1, x1); x++)
            {
                float dx = x - c.x, dy = y - c.y;
                if (dx * dx + dy * dy <= r2) img[y * W + x] = 255;
            }
    }

    static void Line(byte[] img, int W, int H, Vector2 a, Vector2 b, float r)
    {
        float len = Vector2.Distance(a, b);
        int n = Mathf.Max(1, Mathf.CeilToInt(len / 0.5f));
        for (int i = 0; i <= n; i++) Disk(img, W, H, Vector2.Lerp(a, b, (float)i / n), r);
    }

    // PNG エンコーダは下の行から並べる前提なので、上下を入れ替えて渡す
    static byte[] Encode(byte[] img, int W, int H)
    {
        var flipped = new byte[img.Length];
        for (int y = 0; y < H; y++) System.Array.Copy(img, y * W, flipped, (H - 1 - y) * W, W);
        return ImageConversion.EncodeArrayToPNG(flipped, GraphicsFormat.R8_UNorm, (uint)W, (uint)H);
    }

    static string N(double d) => d.ToString("G9", CultureInfo.InvariantCulture);
    static string Q(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
}
