using System.Collections;
using System.IO;
using UnityEngine;

/// <summary>
/// Slide capture helper (2026-10-06). Records frames from its own camera while the robot is in a
/// candidate physics scene (physics_<scene>_<cid>). Added at runtime only; not part of the pipeline.
/// Time.captureFramerate = 50 so one rendered frame = one physics step (dt 0.02).
/// </summary>
public class StepFrameRecorder : MonoBehaviour
{
    public Transform robotRoot;
    public string outDir = "";
    public string[] onlyScenes = new string[0];   // empty = all physics_* scenes
    public int every = 2;
    public int width = 1280, height = 720;
    public Vector3 camPos = new Vector3(0.75f, 1.45f, 1.05f);
    public Vector3 lookAt = new Vector3(0f, 0.9f, 0.25f);
    public float fov = 40f;
    public bool recordHome = false;
    public int frames;
    public Component episodeRecorder;   // if set, a green disc marks its goalCenter
    public float tableTopY = 0.7735f;
    GameObject goalMarker;

    Camera cam; RenderTexture rt; Texture2D tex; int counter;

    public void Init()
    {
        var go = new GameObject("_SlideCam");
        cam = go.AddComponent<Camera>();
        cam.fieldOfView = fov;
        cam.nearClipPlane = 0.02f;
        go.transform.position = camPos;
        go.transform.LookAt(lookAt);
        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        cam.targetTexture = rt;
        tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        Time.captureFramerate = 50;
        StartCoroutine(Loop());
    }

    public void Aim(Vector3 pos, Vector3 target, float f)
    {
        cam.transform.position = pos; cam.transform.LookAt(target); cam.fieldOfView = f;
    }

    bool Match(string s)
    {
        if (!s.StartsWith("physics_")) return recordHome;
        if (onlyScenes == null || onlyScenes.Length == 0) return true;
        foreach (var o in onlyScenes) if (s.EndsWith(o)) return true;
        return false;
    }

    IEnumerator Loop()
    {
        while (enabled)
        {
            yield return new WaitForEndOfFrame();
            UpdateGoal();
            if (robotRoot == null || string.IsNullOrEmpty(outDir)) continue;
            string sn = robotRoot.gameObject.scene.name;
            if (!Match(sn)) continue;
            counter++;
            if (counter % every != 0) continue;
            Grab();
            string d = Path.Combine(outDir, sn);
            Directory.CreateDirectory(d);
            int idx = Directory.GetFiles(d, "*.jpg").Length;
            File.WriteAllBytes(Path.Combine(d, $"f_{idx:D5}.jpg"), tex.EncodeToJPG(92));
            frames++;
        }
    }

    void UpdateGoal()
    {
        if (episodeRecorder == null) return;
        if (goalMarker == null)
        {
            goalMarker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            goalMarker.name = "_GoalMarker";
            Destroy(goalMarker.GetComponent<Collider>());
            var m = new Material(Shader.Find("HDRP/Unlit"));
            m.SetColor("_UnlitColor", new Color(0.2f, 0.75f, 0.35f, 1f));
            goalMarker.GetComponent<MeshRenderer>().sharedMaterial = m;
        }
        var f = episodeRecorder.GetType().GetField("goalCenter");
        var r = episodeRecorder.GetType().GetField("goalRadius");
        Vector3 g = (Vector3)f.GetValue(episodeRecorder);
        float rad = (float)r.GetValue(episodeRecorder);
        goalMarker.transform.position = new Vector3(g.x, tableTopY + 0.0005f, g.z);
        goalMarker.transform.localScale = new Vector3(rad * 2f, 0.0005f, rad * 2f);
    }

    void Grab()
    {
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply(false);
        RenderTexture.active = prev;
    }

    public void SaveStill(string path)
    {
        Grab();
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, path.EndsWith(".png") ? tex.EncodeToPNG() : tex.EncodeToJPG(95));
    }

    void OnDestroy() { Time.captureFramerate = 0; }
}
