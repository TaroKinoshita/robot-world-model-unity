using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// Week 9 video helper (2026-10-23). Replays a recorded episode (executed/executed_trajectory.csv)
/// kinematically and renders frames from its own camera. Not part of the data pipeline.
///
/// Usage (Play mode, with EpisodeRecorder disabled so nothing is generated):
///   1. Play → wait ~1 s so RobotInitialPose closes the fingers
///   2. TrajectoryReplayRenderer.Freeze()            // disables the articulation, keeps finger pose
///   3. TrajectoryReplayRenderer.SetupCamera(...) / SetGoal(...)
///   4. TrajectoryReplayRenderer.RenderEpisode(csv, outDir, stride)  // one JPG per stride rows
/// The robot links and the objects are set from the recorded world poses, so the video shows
/// exactly what is in the data (no new simulation).
/// </summary>
public static class TrajectoryReplayRenderer
{
    static readonly string[] Links = { "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
                                       "wrist_1_link", "wrist_2_link", "wrist_3_link", "tool0" };
    static Camera cam; static RenderTexture rt; static Texture2D tex;
    static GameObject goalMarker;
    static Dictionary<string, Transform> linkT;

    public static string Freeze()
    {
        var robot = GameObject.Find("ur5e_with_gripper");
        linkT = new Dictionary<string, Transform>();
        foreach (var t in robot.GetComponentsInChildren<Transform>(true))
            foreach (var n in Links) if (t.name == n && !linkT.ContainsKey(n)) linkT[n] = t;
        // Freeze every articulation body (root first) so transforms can be set directly.
        var abs = robot.GetComponentsInChildren<ArticulationBody>(true);
        foreach (var ab in abs) ab.enabled = false;
        foreach (var rb in GameObject.Find("Objects").GetComponentsInChildren<Rigidbody>(true)) rb.isKinematic = true;
        return "links " + linkT.Count + ", articulation bodies disabled " + abs.Length;
    }

    public static void SetupCamera(Vector3 pos, Vector3 lookAt, float fov, int w, int h)
    {
        if (cam == null)
        {
            var go = new GameObject("_ReplayCam");
            cam = go.AddComponent<Camera>();
            cam.nearClipPlane = 0.02f;
            // Frames are teleported poses, so turn off motion blur / motion vectors / TAA (they smear).
            var hd = go.AddComponent<UnityEngine.Rendering.HighDefinition.HDAdditionalCameraData>();
            hd.antialiasing = UnityEngine.Rendering.HighDefinition.HDAdditionalCameraData.AntialiasingMode.None;
            hd.customRenderingSettings = true;
            var F = UnityEngine.Rendering.HighDefinition.FrameSettingsField.MotionBlur;
            var M = UnityEngine.Rendering.HighDefinition.FrameSettingsField.MotionVectors;
            hd.renderingPathCustomFrameSettingsOverrideMask.mask[(uint)F] = true;
            hd.renderingPathCustomFrameSettings.SetEnabled(F, false);
            hd.renderingPathCustomFrameSettingsOverrideMask.mask[(uint)M] = true;
            hd.renderingPathCustomFrameSettings.SetEnabled(M, false);
        }
        cam.transform.position = pos; cam.transform.LookAt(lookAt); cam.fieldOfView = fov;
        if (rt == null || rt.width != w || rt.height != h)
        {
            rt = new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32);
            rt.antiAliasing = 1;
            tex = new Texture2D(w, h, TextureFormat.RGB24, false);
        }
        cam.targetTexture = rt;
        cam.enabled = false;   // rendered manually
    }

    /// <summary>Camera matrices for projecting world points in Python (OpenCV-style pixel coords).</summary>
    public static string CameraJson()
    {
        Matrix4x4 V = cam.worldToCameraMatrix, Pm = cam.projectionMatrix;
        var sb = new System.Text.StringBuilder("{\"width\":" + rt.width + ",\"height\":" + rt.height + ",\"view\":[");
        for (int i = 0; i < 16; i++) sb.Append((i > 0 ? "," : "") + V[i / 4, i % 4].ToString("R", CultureInfo.InvariantCulture));
        sb.Append("],\"proj\":[");
        for (int i = 0; i < 16; i++) sb.Append((i > 0 ? "," : "") + Pm[i / 4, i % 4].ToString("R", CultureInfo.InvariantCulture));
        sb.Append("]}");
        return sb.ToString();
    }

    public static void SetGoal(Vector3 centerXZ, float radius, float tableTopY, bool show)
    {
        if (goalMarker == null)
        {
            goalMarker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            goalMarker.name = "_GoalMarker";
            Object.Destroy(goalMarker.GetComponent<Collider>());
            var m = new Material(Shader.Find("HDRP/Unlit"));
            m.SetColor("_UnlitColor", new Color(0.2f, 0.75f, 0.35f, 1f));
            goalMarker.GetComponent<MeshRenderer>().sharedMaterial = m;
        }
        goalMarker.SetActive(show);
        goalMarker.transform.position = new Vector3(centerXZ.x, tableTopY + 0.0005f, centerXZ.z);
        goalMarker.transform.localScale = new Vector3(radius * 2f, 0.0005f, radius * 2f);
    }

    public static void ShowSecondary(bool show)
    {
        var so = GameObject.Find("Objects").transform.Find("SecondObject");
        so.gameObject.SetActive(show);
    }

    public static void SetSecondaryHeight(float h)
    {
        var so = GameObject.Find("Objects").transform.Find("SecondObject");
        var s = so.localScale; // capsule: height = 2 * scale.y, radius 0.048 → scale.x = 0.096
        so.localScale = new Vector3(s.x, h / 2f, s.z);
    }

    static float F(string s) { return float.Parse(s, CultureInfo.InvariantCulture); }

    /// <summary>Renders every `stride`-th row of the CSV (plus the last row). Returns number of frames.</summary>
    public static int RenderEpisode(string csvPath, string outDir, int stride)
    {
        Directory.CreateDirectory(outDir);
        var lines = File.ReadAllLines(csvPath);
        var head = lines[0].Split(',');
        var col = new Dictionary<string, int>();
        for (int i = 0; i < head.Length; i++) col[head[i]] = i;
        var objs = GameObject.Find("Objects").transform;
        var target = objs.Find("Target");
        var second = objs.Find("SecondObject");
        bool hasSecond = col.ContainsKey("SecondObject_x");
        int n = 0;
        int last = lines.Length - 1;
        for (int r = 1; r <= last; r++)
        {
            if ((r - 1) % stride != 0 && r != last) continue;
            var v = lines[r].Split(',');
            foreach (var name in Links)
            {
                var t = linkT[name];
                t.position = new Vector3(F(v[col[name + "_x"]]), F(v[col[name + "_y"]]), F(v[col[name + "_z"]]));
                t.rotation = new Quaternion(F(v[col[name + "_rx"]]), F(v[col[name + "_ry"]]), F(v[col[name + "_rz"]]), F(v[col[name + "_rw"]]));
            }
            target.position = new Vector3(F(v[col["Target_x"]]), F(v[col["Target_y"]]), F(v[col["Target_z"]]));
            target.rotation = new Quaternion(F(v[col["Target_rx"]]), F(v[col["Target_ry"]]), F(v[col["Target_rz"]]), F(v[col["Target_rw"]]));
            if (hasSecond && second.gameObject.activeSelf)
            {
                second.position = new Vector3(F(v[col["SecondObject_x"]]), F(v[col["SecondObject_y"]]), F(v[col["SecondObject_z"]]));
                second.rotation = new Quaternion(F(v[col["SecondObject_rx"]]), F(v[col["SecondObject_ry"]]), F(v[col["SecondObject_rz"]]), F(v[col["SecondObject_rw"]]));
            }
            cam.Render();
            var prev = RenderTexture.active;
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply(false);
            RenderTexture.active = prev;
            File.WriteAllBytes(Path.Combine(outDir, "f_" + n.ToString("D5") + ".jpg"), tex.EncodeToJPG(93));
            n++;
        }
        return n;
    }

    /// <summary>Sets the scene to one row (e.g. the first) without rendering.</summary>
    public static void Pose(string csvPath, int row)
    {
        var lines = File.ReadAllLines(csvPath);
        var head = lines[0].Split(',');
        var col = new Dictionary<string, int>();
        for (int i = 0; i < head.Length; i++) col[head[i]] = i;
        var v = lines[Mathf.Clamp(row, 1, lines.Length - 1)].Split(',');
        foreach (var name in Links)
        {
            var t = linkT[name];
            t.position = new Vector3(F(v[col[name + "_x"]]), F(v[col[name + "_y"]]), F(v[col[name + "_z"]]));
            t.rotation = new Quaternion(F(v[col[name + "_rx"]]), F(v[col[name + "_ry"]]), F(v[col[name + "_rz"]]), F(v[col[name + "_rw"]]));
        }
        var target = GameObject.Find("Objects").transform.Find("Target");
        target.position = new Vector3(F(v[col["Target_x"]]), F(v[col["Target_y"]]), F(v[col["Target_z"]]));
        target.rotation = new Quaternion(F(v[col["Target_rx"]]), F(v[col["Target_ry"]]), F(v[col["Target_rz"]]), F(v[col["Target_rw"]]));
    }

    public static void SaveStill(string path)
    {
        cam.Render();
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
        tex.Apply(false);
        RenderTexture.active = prev;
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, tex.EncodeToJPG(93));
    }
}
