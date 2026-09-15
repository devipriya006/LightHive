using UnityEngine;

public class LightHiveLightingController : MonoBehaviour
{
    public Light[] discoLights;
    public Renderer ledWall;

    public Color stageColor = Color.blue;
    public float brightness = 40f;
    public float transitionSpeed = 5f;

    [Header("Beat Effect")]
    public bool beatDetected = false;
    public float beatFlashIntensity = 80f;
    public float beatFlashDuration = 0.12f;

    private float beatTimer = 0f;

    void Update()
    {
        if (beatDetected)
        {
            beatTimer = beatFlashDuration;
            beatDetected = false;
        }

        if (beatTimer > 0)
        {
            beatTimer -= Time.deltaTime;

            for (int i = 0; i < discoLights.Length; i++)
            {
                discoLights[i].color = stageColor;
                discoLights[i].intensity = beatFlashIntensity;
            }
            if (ledWall != null)
            {
                ledWall.material.color = stageColor;
            }
        }
        else
        {
            for (int i = 0; i < discoLights.Length; i++)
            {
                discoLights[i].color = Color.Lerp(
                    discoLights[i].color,
                    stageColor,
                    transitionSpeed * Time.deltaTime
                );

                discoLights[i].intensity = Mathf.Lerp(
                    discoLights[i].intensity,
                    brightness,
                    transitionSpeed * Time.deltaTime
                );
            }

            if (ledWall != null)
            {
                ledWall.material.color = Color.Lerp(
                    ledWall.material.color,
                    stageColor,
                    transitionSpeed * Time.deltaTime
                );
            }
        }
    }
}