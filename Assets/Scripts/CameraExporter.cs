using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

/// <summary>
/// TODO5-1: カメラの K / projection matrix / world-to-camera matrix を JSON に書き出す。
/// Play 開始から framesBeforeExport フレーム後に自動で書き出す(Inspector 右クリック > Export Now でも可)。
/// 出力先: &lt;プロジェクト&gt;/CaptureLogs/camera_YYYYMMDD_HHMMSS.json
///
/// 座標系:
///   world      = Unity world(左手系、Y 上、メートル)
///   camera(cv) = OpenCV 規約(x 右、y 下、z 前、メートル)
///   pixel      = OpenCV 規約((0,0) は左上ピクセルの中心、u 右、v 下)
/// 投影: [u, v, 1]^T ∝ K · (T_world_to_cam_cv · [X, Y, Z, 1]^T)[:3]
/// </summary>
public class CameraExporter : MonoBehaviour
{
    [Tooltip("PerceptionCamera が付いているカメラ。空ならこのGameObjectのCamera")]
    public Camera targetCamera;

    [Tooltip("投影チェック用の直方体(Cube)。0 番目に Target。空なら \"Target\" を自動で探す")]
    public Transform[] checkObjects;

    public int expectedWidth = 256;
    public int expectedHeight = 256;

    [Tooltip("Play 開始から何フレーム待って書き出すか")]
    public int framesBeforeExport = 5;

    IEnumerator Start()
    {
        for (int i = 0; i < framesBeforeExport; i++) yield return null;
        Export();
    }

    [ContextMenu("Export Now")]
    public void Export()
    {
        Camera cam = targetCamera != null ? targetCamera : GetComponent<Camera>();
        if (cam == null) { Debug.LogError("[CameraExporter] Camera が未設定"); return; }
        if (cam.orthographic) { Debug.LogError("[CameraExporter] Orthographic は非対応"); return; }

        int W = cam.pixelWidth;
        int H = cam.pixelHeight;
        if (W != expectedWidth || H != expectedHeight)
            Debug.LogWarning($"[CameraExporter] 解像度 {W}x{H} が期待値 {expectedWidth}x{expectedHeight} と違う。Game View を capture_256 にして Play 中に書き出す");
        if (cam.usePhysicalProperties)
            Debug.LogWarning("[CameraExporter] Physical Camera がオン。Lens Shift / Sensor Size が K に効くので確認する");

        // Unity の行列
        Matrix4x4 P = cam.nonJitteredProjectionMatrix; // OpenGL 規約の projection
        Matrix4x4 V = cam.worldToCameraMatrix;         // Unity world → view(右手系、-z が前)

        // view(OpenGL) → OpenCV:y と z を反転
        Matrix4x4 T = Matrix4x4.Scale(new Vector3(1f, -1f, -1f)) * V;

        // projection から K を作る(lens shift があっても正しく出る形)
        double fx = P.m00 * W / 2.0;
        double fy = P.m11 * H / 2.0;
        double cx = W / 2.0 * (1.0 - P.m02) - 0.5;
        double cy = H / 2.0 * (1.0 + P.m12) - 0.5;

        var f = new List<string>();
        f.Add(KV("schema", Q("camera_v1")));
        f.Add(KV("created_at", Q(DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture))));
        f.Add(KV("unity_version", Q(Application.unityVersion)));
        f.Add(KV("camera_name", Q(cam.name)));
        f.Add(KV("width", W.ToString(CultureInfo.InvariantCulture)));
        f.Add(KV("height", H.ToString(CultureInfo.InvariantCulture)));
        f.Add(KV("fov_vertical_deg", N(cam.fieldOfView)));
        f.Add(KV("near", N(cam.nearClipPlane)));
        f.Add(KV("far", N(cam.farClipPlane)));
        f.Add(KV("physical_camera", cam.usePhysicalProperties ? "true" : "false"));
        f.Add(KV("position_world", Vec(cam.transform.position)));
        f.Add(KV("rotation_world_xyzw", Quat(cam.transform.rotation)));
        f.Add(KV("euler_world_deg", Vec(cam.transform.eulerAngles)));
        f.Add(KV("world_frame", Q("Unity world: left-handed, Y-up, meters")));
        f.Add(KV("camera_frame_cv", Q("OpenCV: x right, y down, z forward, meters")));
        f.Add(KV("pixel_convention", Q("OpenCV: (0,0) is the center of the top-left pixel, u right, v down")));
        f.Add(KV("K", $"[[{N(fx)}, 0, {N(cx)}], [0, {N(fy)}, {N(cy)}], [0, 0, 1]]"));
        f.Add(KV("T_world_to_cam_cv", Mat(T)));
        f.Add(KV("unity_projection_matrix_gl", Mat(P)));
        f.Add(KV("unity_world_to_camera_matrix", Mat(V)));

        // 投影チェック用の物体(直方体の 8 頂点を world 座標で出す)
        // checkObjects が空なら "Target" を自動で探す
        if (checkObjects == null || checkObjects.Length == 0)
        {
            var targetGo = GameObject.Find("Target");
            if (targetGo != null) checkObjects = new[] { targetGo.transform };
        }

        var objs = new List<string>();
        if (checkObjects != null)
        {
            foreach (var t in checkObjects)
            {
                if (t == null) continue;
                string corners = "null";
                var mf = t.GetComponent<MeshFilter>();
                if (mf != null && mf.sharedMesh != null)
                {
                    Bounds b = mf.sharedMesh.bounds;
                    var cs = new List<string>();
                    for (int i = 0; i < 8; i++)
                    {
                        var s = new Vector3((i & 1) == 0 ? -1f : 1f, (i & 2) == 0 ? -1f : 1f, (i & 4) == 0 ? -1f : 1f);
                        cs.Add(Vec(t.TransformPoint(b.center + Vector3.Scale(b.extents, s))));
                    }
                    corners = "[" + string.Join(", ", cs) + "]";
                }
                objs.Add($"    {{\"name\": {Q(t.name)}, \"position_world\": {Vec(t.position)}, " +
                         $"\"rotation_world_xyzw\": {Quat(t.rotation)}, \"corners_world\": {corners}}}");
            }
        }
        f.Add(KV("check_objects", "[\n" + string.Join(",\n", objs) + "\n  ]"));

        string json = "{\n" + string.Join(",\n", f) + "\n}\n";

        string dir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "CaptureLogs"));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"camera_{DateTime.Now:yyyyMMdd_HHmmss}.json");
        File.WriteAllText(path, json, new UTF8Encoding(false));

        Debug.Log($"[CameraExporter] 書き出し: {path}\n" +
                  $"K: fx={fx:F3} fy={fy:F3} cx={cx:F3} cy={cy:F3} / {W}x{H}");
    }

    // ---- JSON ヘルパー ----
    static string N(double d) => d.ToString("G10", CultureInfo.InvariantCulture);
    static string Q(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    static string KV(string k, string v) => $"  \"{k}\": {v}";
    static string Vec(Vector3 v) => $"[{N(v.x)}, {N(v.y)}, {N(v.z)}]";
    static string Quat(Quaternion q) => $"[{N(q.x)}, {N(q.y)}, {N(q.z)}, {N(q.w)}]";
    static string Mat(Matrix4x4 m)
    {
        var rows = new string[4];
        for (int r = 0; r < 4; r++)
            rows[r] = $"[{N(m[r, 0])}, {N(m[r, 1])}, {N(m[r, 2])}, {N(m[r, 3])}]";
        return "[" + string.Join(", ", rows) + "]";
    }
}
