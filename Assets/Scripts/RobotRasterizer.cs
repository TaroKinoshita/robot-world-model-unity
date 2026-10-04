using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// P4-B: 計画した関節角でロボットの見た目のメッシュを CPU で描いて、マスクと depth を作る。
/// GPU の描画(HDRP)を使わないので、計画の段階で描けて、カメラの幾何だけで結果が決まる(検証しやすい)。
/// - 画素の規約は ActionRepresentations.Project と同じ(OpenCV、左上の画素の中心が (0, 0))
/// - depth = カメラ座標の z(カメラの前方向の距離、m)。画素の中心で、三角形の上を 1/z で線形補間する
/// - EEF = グリッパー(chain の外のリンク: robotiq_arg2f_base_link から先)、Full-body = ロボット全体
/// </summary>
public class RobotRasterizer
{
    class Part { public int link; public bool eef; public Vector3[] v; public int[] t; }

    readonly List<Part> parts = new List<Part>();
    ArmKinematics kin;
    int toolIndex;
    public int TriangleCount { get; private set; }
    public int EefTriangleCount { get; private set; }

    public static RobotRasterizer Build(ArmKinematics kin, Transform robotRoot)
    {
        var r = new RobotRasterizer { kin = kin };
        var names = kin.LinkNames;
        r.toolIndex = names.Length - 1;
        var linkT = new Transform[names.Length];
        foreach (var ab in robotRoot.GetComponentsInChildren<ArticulationBody>(true))
        {
            int i = System.Array.IndexOf(names, ab.name);
            if (i >= 0 && linkT[i] == null) linkT[i] = ab.transform;
        }
        foreach (var mf in robotRoot.GetComponentsInChildren<MeshFilter>(false))
        {
            var rend = mf.GetComponent<Renderer>();
            if (rend == null || !rend.enabled || mf.sharedMesh == null) continue;
            var owner = mf.GetComponentInParent<ArticulationBody>();
            int li = owner != null ? System.Array.IndexOf(linkT, owner.transform) : -1;
            bool eef = li < 0;                 // chain の外 = グリッパー(tool に固定、指は初期の角度のまま)
            Transform frame = eef ? linkT[r.toolIndex] : linkT[li];
            var m = mf.sharedMesh;
            var src = m.vertices;
            var v = new Vector3[src.Length];
            for (int k = 0; k < src.Length; k++) v[k] = frame.InverseTransformPoint(mf.transform.TransformPoint(src[k]));
            var t = m.triangles;
            r.parts.Add(new Part { link = eef ? r.toolIndex : li, eef = eef, v = v, t = t });
            r.TriangleCount += t.Length / 3;
            if (eef) r.EefTriangleCount += t.Length / 3;
        }
        return r;
    }

    /// <summary>depth(W*H、行 0 = 画像の上、0 = 何も無い)に描く</summary>
    public void Render(float[] q, Camera cam, int W, int H, bool eefOnly, float[] depth)
    {
        int L = kin.LinkNames.Length;
        var pos = new Vector3[L]; var rot = new Quaternion[L];
        kin.LinkPoses(q, pos, rot);
        Matrix4x4 V = cam.worldToCameraMatrix, P = cam.projectionMatrix;
        Matrix4x4 PV = P * V;
        float near = cam.nearClipPlane;
        System.Array.Clear(depth, 0, depth.Length);
        var zbuf = new float[W * H];
        for (int i = 0; i < zbuf.Length; i++) zbuf[i] = float.PositiveInfinity;

        foreach (var part in parts)
        {
            if (eefOnly && !part.eef) continue;
            var p = pos[part.link]; var rr = rot[part.link];
            int n = part.v.Length;
            var sx = new float[n]; var sy = new float[n]; var sz = new float[n];
            for (int k = 0; k < n; k++)
            {
                Vector3 w = p + rr * part.v[k];
                Vector3 c = V.MultiplyPoint3x4(w);
                float z = -c.z;
                Vector4 clip = PV * new Vector4(w.x, w.y, w.z, 1f);
                float iw = 1f / clip.w;
                sx[k] = (clip.x * iw + 1f) * 0.5f * W - 0.5f;
                sy[k] = H - (clip.y * iw + 1f) * 0.5f * H - 0.5f;
                sz[k] = z;
            }
            var t = part.t;
            for (int k = 0; k < t.Length; k += 3)
            {
                int a = t[k], b = t[k + 1], c = t[k + 2];
                if (sz[a] <= near || sz[b] <= near || sz[c] <= near) continue;
                Tri(sx[a], sy[a], sz[a], sx[b], sy[b], sz[b], sx[c], sy[c], sz[c], W, H, zbuf);
            }
        }
        for (int i = 0; i < zbuf.Length; i++) depth[i] = float.IsPositiveInfinity(zbuf[i]) ? 0f : zbuf[i];
    }

    static void Tri(float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2, int W, int H, float[] zbuf)
    {
        float area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if (Mathf.Abs(area) < 1e-12f) return;
        int xmin = Mathf.Max(0, Mathf.CeilToInt(Mathf.Min(x0, Mathf.Min(x1, x2))));
        int xmax = Mathf.Min(W - 1, Mathf.FloorToInt(Mathf.Max(x0, Mathf.Max(x1, x2))));
        int ymin = Mathf.Max(0, Mathf.CeilToInt(Mathf.Min(y0, Mathf.Min(y1, y2))));
        int ymax = Mathf.Min(H - 1, Mathf.FloorToInt(Mathf.Max(y0, Mathf.Max(y1, y2))));
        if (xmin > xmax || ymin > ymax) return;
        float inv = 1f / area;
        float iz0 = 1f / z0, iz1 = 1f / z1, iz2 = 1f / z2;
        for (int y = ymin; y <= ymax; y++)
            for (int x = xmin; x <= xmax; x++)
            {
                float w0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) * inv;
                float w1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) * inv;
                float w2 = 1f - w0 - w1;
                if (w0 < 0f || w1 < 0f || w2 < 0f) continue;
                float z = 1f / (w0 * iz0 + w1 * iz1 + w2 * iz2);
                int idx = y * W + x;
                if (z < zbuf[idx]) zbuf[idx] = z;
            }
    }
}
