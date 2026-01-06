import os
import time
import requests
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from tqdm import tqdm

class SatelliteImageFetcher:
    def __init__(self, mapbox_token, image_size="256x256", zoom=16, map_style="satellite-v9"):
        self.mapbox_token = mapbox_token
        self.image_size = image_size
        self.zoom = zoom
        self.map_style = map_style

    def fetch_image(self, lat, lon, save_path, retries=3, sleep_time=0.3):
        """
        Fetch a satellite image using Mapbox Static Images API.
        If the image already exists, it will skip downloading.
        Includes retry + timeout handling.
        """
        if os.path.exists(save_path):
            return True

        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/{self.map_style}/static/"
            f"{lon},{lat},{self.zoom}/{self.image_size}"
            f"?access_token={self.mapbox_token}"
        )

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    time.sleep(sleep_time)
                    return True
                else:
                    print(f"[Warning] Status {response.status_code} for ({lat}, {lon})")
            except (ReadTimeout, ConnectionError) as e:
                print(f"[Timeout] Attempt {attempt}/{retries} for ({lat}, {lon})")
                time.sleep(2)  # backoff

        print(f"[Error] Failed to fetch image for ({lat}, {lon}) after {retries} attempts")
        return False

    def fetch_images_from_dataframe(self, df, lat_col="lat", lon_col="long", save_dir="images"):
        """
        Fetch satellite images for all rows in a dataframe.
        Returns a list of image paths (None if download failed).
        """
        image_paths = []

        os.makedirs(save_dir, exist_ok=True)
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(save_dir, f"{idx}.png")
            success = self.fetch_image(row[lat_col], row[lon_col], img_path)
            image_paths.append(img_path if success else None)

        df["image_path"] = image_paths
        return df

if __name__ == "__main__":
    MAPBOX_TOKEN = "pk.eyJ1IjoicG1lZW5ha3NoaSIsImEiOiJjbWp0bWk5d2IyandnM2VzZTlobGs4eHR6In0.P5oWxqmEIdIjwpJO8QwBBQ"
    TRAIN_CSV = "Downloads/train(1).xlsx"
    TEST_CSV = "Downloads/test2.xlsx"
    TRAIN_IMG_DIR = "data_cdc/satellite_images/train"
    TEST_IMG_DIR = "data_cdc/satellite_images/test"

    # Load data
    train_df = pd.read_excel(TRAIN_CSV).reset_index(drop=True)
    test_df = pd.read_excel(TEST_CSV).reset_index(drop=True)

    fetcher = SatelliteImageFetcher(MAPBOX_TOKEN)

    # Fetch images
    train_df = fetcher.fetch_images_from_dataframe(train_df, save_dir=TRAIN_IMG_DIR)
    test_df = fetcher.fetch_images_from_dataframe(test_df, save_dir=TEST_IMG_DIR)

    # Save updated dataframes with image paths
    train_df.to_excel("train_with_images.xlsx", index=False)
    test_df.to_excel("test_with_images.xlsx", index=False)
