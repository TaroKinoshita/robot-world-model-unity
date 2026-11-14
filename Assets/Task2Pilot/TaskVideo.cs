using System.IO;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

/// <summary>
/// 説明用の動画のコマを撮る（2・3 つ目のタスクの recorder から呼ぶ）。videoDir が空なら何もしない（データ作りには影響しない）。
/// 物理の 1 step ごとに、メインカメラを別の RenderTexture（videoW × videoH）に描いて PNG に保存する。
/// </summary>
public static class TaskVideo
{
    static string dir;
    static int n;
    static RenderTexture rt;
    static Texture2D tex;
    static Camera vcam;
    public static int W = 960, H = 720;

    public static void Begin(string videoDir, string cands, string sid, string cid)
    {
        dir = null; n = 0;
        if (string.IsNullOrEmpty(videoDir) || string.IsNullOrEmpty(cands) || !cands.Contains(cid)) return;
        dir = Path.Combine(Path.GetFullPath(Path.Combine(Application.dataPath, "..", videoDir)), sid + "_" + cid);
        Directory.CreateDirectory(dir);
    }

    public static void End() { dir = null; }

    public static void Frame(Camera cam)
    {
        if (dir == null || cam == null) return;
        if (rt == null) { rt = new RenderTexture(W, H, 24, RenderTextureFormat.ARGB32) { name = "TaskVideo_RT" }; tex = new Texture2D(W, H, TextureFormat.RGB24, false); }
        if (vcam == null)
        {
            // データ用のカメラとは別のカメラで撮る（同じカメラで RT を切り替えると、履歴が混ざってモーションブラーが出たため）
            var go = new GameObject("TaskVideoCam");
            vcam = go.AddComponent<Camera>();
            vcam.CopyFrom(cam); vcam.enabled = false; vcam.targetTexture = rt;
            var hd = go.AddComponent<HDAdditionalCameraData>();
            hd.antialiasing = HDAdditionalCameraData.AntialiasingMode.None;
            hd.customRenderingSettings = true;
            hd.renderingPathCustomFrameSettingsOverrideMask.mask[(uint)FrameSettingsField.MotionBlur] = true;
            hd.renderingPathCustomFrameSettings.SetEnabled(FrameSettingsField.MotionBlur, false);
        }
        vcam.transform.SetPositionAndRotation(cam.transform.position, cam.transform.rotation);
        vcam.Render();
        var pa = RenderTexture.active; RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, W, H), 0, 0); tex.Apply();
        RenderTexture.active = pa;
        File.WriteAllBytes(Path.Combine(dir, $"f_{n++:D5}.png"), tex.EncodeToPNG());
    }
}
