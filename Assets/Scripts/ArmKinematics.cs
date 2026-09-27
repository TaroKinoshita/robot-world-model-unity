using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// ArticulationBody の anchor 情報から、ロボットアームの順運動学(FK)と数値 IK を計算する。
/// 関節角は Unity の jointPosition と同じ単位(rad)・同じ原点。回転の符号は JointSign(実機の姿勢と比べて検証で決める)。
/// TCP(手先の基準点)= 2 つの指パッドの中点。
/// </summary>
public class ArmKinematics
{
    class Node
    {
        public string name;
        public int q = -1;                      // 関節角の index(-1 = 固定関節)
        public Vector3 pAnchorPos, anchorPos;
        public Quaternion pAnchorRot, anchorRot;
    }

    readonly List<Node> nodes = new List<Node>();   // root → tool の順(一本鎖)
    Vector3 rootPos;
    Quaternion rootRot;

    public string[] JointNames { get; private set; }
    public ArticulationBody[] JointBodies { get; private set; }
    public Transform Tool { get; private set; }
    public float[] LowerLimit { get; private set; }  // rad
    public float[] UpperLimit { get; private set; }  // rad
    public Vector3 TcpLocal { get; private set; }       // tool リンク座標での TCP
    public Vector3 ApproachLocal { get; private set; }  // tool リンク座標でのグリッパーの向き(tool 原点 → TCP)
    public Vector3 OpeningLocal { get; private set; }   // tool リンク座標での指の開閉方向(pad A → pad B)
    public float FingerHalfSpan { get; private set; }   // TCP から指パッドまでの距離
    public float JointSign = 1f;
    public int Dof => JointNames.Length;

    public static ArmKinematics Build(Transform robotRoot, string toolLinkName, string[] armJointNames,
                                      string padAName, string padBName, out string error)
    {
        error = null;
        var k = new ArmKinematics();

        ArticulationBody tool = null;
        foreach (var ab in robotRoot.GetComponentsInChildren<ArticulationBody>(true))
            if (ab.name == toolLinkName) { tool = ab; break; }
        if (tool == null) { error = $"tool link '{toolLinkName}' が見つからない"; return null; }

        var chain = new List<ArticulationBody>();
        for (var b = tool; b != null; b = ParentBody(b)) chain.Add(b);
        chain.Reverse();
        if (!chain[0].isRoot) { error = "root の ArticulationBody まで辿れない"; return null; }

        int n = armJointNames.Length;
        k.JointNames = armJointNames;
        k.JointBodies = new ArticulationBody[n];
        k.LowerLimit = new float[n];
        k.UpperLimit = new float[n];
        k.rootPos = chain[0].transform.position;
        k.rootRot = chain[0].transform.rotation;

        for (int i = 0; i < chain.Count; i++)
        {
            var b = chain[i];
            var node = new Node
            {
                name = b.name,
                pAnchorPos = b.parentAnchorPosition, pAnchorRot = b.parentAnchorRotation,
                anchorPos = b.anchorPosition, anchorRot = b.anchorRotation
            };
            if (i > 0 && b.dofCount > 0)
            {
                int qi = Array.IndexOf(armJointNames, b.name);
                if (qi < 0) { error = $"chain 上の可動関節 '{b.name}' が armJointNames に無い"; return null; }
                if (b.jointType != ArticulationJointType.RevoluteJoint) { error = $"'{b.name}' は回転関節ではない"; return null; }
                node.q = qi;
                k.JointBodies[qi] = b;
                var d = b.xDrive;
                k.LowerLimit[qi] = d.lowerLimit * Mathf.Deg2Rad;
                k.UpperLimit[qi] = d.upperLimit * Mathf.Deg2Rad;
            }
            k.nodes.Add(node);
        }
        for (int j = 0; j < n; j++)
            if (k.JointBodies[j] == null) { error = $"'{armJointNames[j]}' が tool までの chain 上に無い"; return null; }
        k.Tool = tool.transform;

        Transform pa = FindChild(robotRoot, padAName), pb = FindChild(robotRoot, padBName);
        if (pa == null || pb == null) { error = $"指パッド '{padAName}' / '{padBName}' が見つからない"; return null; }
        Vector3 mid = (pa.position + pb.position) * 0.5f;
        k.TcpLocal = k.Tool.InverseTransformPoint(mid);
        k.ApproachLocal = k.TcpLocal.normalized;
        Vector3 open = k.Tool.InverseTransformDirection(pb.position - pa.position);
        k.OpeningLocal = Vector3.ProjectOnPlane(open, k.ApproachLocal).normalized;
        k.FingerHalfSpan = (pb.position - pa.position).magnitude * 0.5f;
        return k;
    }

    static ArticulationBody ParentBody(ArticulationBody b)
    {
        for (var p = b.transform.parent; p != null; p = p.parent)
        {
            var ab = p.GetComponent<ArticulationBody>();
            if (ab != null) return ab;
        }
        return null;
    }

    static Transform FindChild(Transform root, string name)
    {
        foreach (var t in root.GetComponentsInChildren<Transform>(true))
            if (t.name == name) return t;
        return null;
    }

    // ---------- FK ----------

    public void ToolPose(float[] q, out Vector3 pos, out Quaternion rot)
    {
        Vector3 p = rootPos;
        Quaternion r = rootRot;
        for (int i = 1; i < nodes.Count; i++)
        {
            var nd = nodes[i];
            Vector3 jp = p + r * nd.pAnchorPos;
            Quaternion jr = r * nd.pAnchorRot;
            if (nd.q >= 0) jr = jr * Quaternion.AngleAxis(JointSign * q[nd.q] * Mathf.Rad2Deg, Vector3.right);
            r = jr * Quaternion.Inverse(nd.anchorRot);
            p = jp - r * nd.anchorPos;
        }
        pos = p;
        rot = r;
    }

    public void TcpPose(float[] q, out Vector3 tcp, out Quaternion toolRot)
    {
        ToolPose(q, out var p, out toolRot);
        tcp = p + toolRot * TcpLocal;
    }

    /// <summary>グリッパーの向き(approach)と指の開閉方向(opening)を world で指定したときの tool の回転</summary>
    public Quaternion ToolRotationFor(Vector3 approachWorld, Vector3 openingWorld)
    {
        Vector3 aL = ApproachLocal, oL = OpeningLocal, cL = Vector3.Cross(aL, oL);
        Vector3 aW = approachWorld.normalized;
        Vector3 oW = Vector3.ProjectOnPlane(openingWorld, aW).normalized;
        Vector3 cW = Vector3.Cross(aW, oW);
        Vector3 Map(Vector3 v) => aW * Vector3.Dot(aL, v) + oW * Vector3.Dot(oL, v) + cW * Vector3.Dot(cL, v);
        return Quaternion.LookRotation(Map(Vector3.forward), Map(Vector3.up));
    }

    // ---------- IK(減衰最小二乗法・数値ヤコビアン) ----------

    public static Vector3 RotVec(Quaternion d)
    {
        if (d.w < 0) { d.x = -d.x; d.y = -d.y; d.z = -d.z; d.w = -d.w; }
        var v = new Vector3(d.x, d.y, d.z);
        float s = v.magnitude;
        if (s < 1e-9f) return 2f * v;
        return v / s * (2f * Mathf.Atan2(s, d.w));
    }

    public bool SolveIK(Vector3 tcpTarget, Quaternion rotTarget, float[] seed, out float[] q,
                        out float posErr, out float angErr, int maxIter = 300, float rotWeight = 0.2f)
    {
        int n = Dof;
        q = (float[])seed.Clone();
        var J = new double[6, n];
        var e = new double[6];
        const float h = 1e-4f;

        for (int it = 0; it < maxIter; it++)
        {
            TcpPose(q, out var tcp, out var rot);
            Vector3 ep = tcpTarget - tcp;
            Vector3 er = RotVec(rotTarget * Quaternion.Inverse(rot));
            if (ep.magnitude < 1e-4f && er.magnitude < 1e-3f) break;

            e[0] = ep.x; e[1] = ep.y; e[2] = ep.z;
            e[3] = er.x * rotWeight; e[4] = er.y * rotWeight; e[5] = er.z * rotWeight;

            for (int j = 0; j < n; j++)
            {
                float old = q[j];
                q[j] = old + h;
                TcpPose(q, out var tcp2, out var rot2);
                q[j] = old;
                Vector3 dp = (tcp2 - tcp) / h;
                Vector3 dr = RotVec(rot2 * Quaternion.Inverse(rot)) / h;
                J[0, j] = dp.x; J[1, j] = dp.y; J[2, j] = dp.z;
                J[3, j] = dr.x * rotWeight; J[4, j] = dr.y * rotWeight; J[5, j] = dr.z * rotWeight;
            }

            var A = new double[6, 6];
            for (int a = 0; a < 6; a++)
                for (int b = 0; b < 6; b++)
                {
                    double s = 0;
                    for (int j = 0; j < n; j++) s += J[a, j] * J[b, j];
                    A[a, b] = s + (a == b ? 1e-4 : 0.0);
                }
            var x = Solve6(A, e);
            if (x == null) break;

            var dq = new float[n];
            float maxStep = 0f;
            for (int j = 0; j < n; j++)
            {
                double s = 0;
                for (int a = 0; a < 6; a++) s += J[a, j] * x[a];
                dq[j] = (float)s;
                maxStep = Mathf.Max(maxStep, Mathf.Abs(dq[j]));
            }
            float scale = maxStep > 0.2f ? 0.2f / maxStep : 1f;
            for (int j = 0; j < n; j++)
                q[j] = Mathf.Clamp(q[j] + dq[j] * scale, LowerLimit[j], UpperLimit[j]);
        }

        TcpPose(q, out var tf, out var rf);
        posErr = (tcpTarget - tf).magnitude;
        angErr = RotVec(rotTarget * Quaternion.Inverse(rf)).magnitude;
        return posErr < 1e-3f && angErr < 1e-2f;
    }

    static double[] Solve6(double[,] A, double[] b)
    {
        int n = 6;
        var M = new double[n, n + 1];
        for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) M[i, j] = A[i, j]; M[i, n] = b[i]; }
        for (int c = 0; c < n; c++)
        {
            int piv = c;
            for (int r = c + 1; r < n; r++) if (Math.Abs(M[r, c]) > Math.Abs(M[piv, c])) piv = r;
            if (Math.Abs(M[piv, c]) < 1e-12) return null;
            if (piv != c) for (int j = 0; j <= n; j++) { var tmp = M[c, j]; M[c, j] = M[piv, j]; M[piv, j] = tmp; }
            for (int r = 0; r < n; r++)
            {
                if (r == c) continue;
                double f = M[r, c] / M[c, c];
                for (int j = c; j <= n; j++) M[r, j] -= f * M[c, j];
            }
        }
        var x = new double[n];
        for (int i = 0; i < n; i++) x[i] = M[i, n] / M[i, i];
        return x;
    }
}
