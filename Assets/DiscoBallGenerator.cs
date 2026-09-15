using UnityEngine;

public class DiscoBallGenerator : MonoBehaviour
{
    public Material tileMaterial;

    public int rows = 10;
    public int columns = 20;

    void Start()
    {
        GenerateTiles();
    }

    void GenerateTiles()
    {
        for (int i = 0; i <= rows; i++)
        {
            float phi = Mathf.PI * i / rows;

            for (int j = 0; j < columns; j++)
            {
                float theta = 2 * Mathf.PI * j / columns;

                Vector3 normal = new Vector3(
                    Mathf.Sin(phi) * Mathf.Cos(theta),
                    Mathf.Cos(phi),
                    Mathf.Sin(phi) * Mathf.Sin(theta)
                );

                GameObject tile = GameObject.CreatePrimitive(PrimitiveType.Cube);

                tile.name = "MirrorTile";
                tile.transform.SetParent(transform);

                tile.transform.localPosition = normal * 0.51f;
                tile.transform.localRotation = Quaternion.LookRotation(normal);
                tile.transform.localScale = new Vector3(0.09f, 0.09f, 0.025f);

                if (tileMaterial != null)
                    tile.GetComponent<Renderer>().material = tileMaterial;

                Destroy(tile.GetComponent<Collider>());
            }
        }
    }
}