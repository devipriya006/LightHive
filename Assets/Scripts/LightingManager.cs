using UnityEngine;

public class LightingManager : MonoBehaviour
{
    public Light[] spotlights;

    [Range(0f, 10f)]
    public float brightness = 5f;

    void Update()
    {
        for (int i = 0; i < spotlights.Length; i++)
        {
            spotlights[i].intensity = brightness;
        }
    }
}