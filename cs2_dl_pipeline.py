import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from awpy import Demo

# --- YAPILANDIRMA VE LOGLAMA ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CS2DataPipeline")

# --- HARİTA SINIRLARI (MAP BOUNDARIES) ---
# awpy'den gelen dünya koordinatlarını 0-1 arasına normalize etmek için kullanılır.
# Not: Bu değerler CS2'nin standart overview/boundary verilerinden alınmıştır.
MAP_BOUNDS = {
    "de_mirage": {"x": (-3230, 1890), "y": (-3407, 1713), "z": (-200, 1000)},
    "de_inferno": {"x": (-2087, 2943), "y": (-1142, 4358), "z": (-200, 1000)},
    "de_dust2": {"x": (-2476, 2127), "y": (-1281, 3819), "z": (-200, 1000)},
    "de_overpass": {"x": (-4831, 503), "y": (-3592, 1820), "z": (-1000, 2000)},
    "de_nuke": {"x": (-3453, 3750), "y": (-4200, 2800), "z": (-1500, 1500)},
    "de_vertigo": {"x": (-3120, 2300), "y": (-2200, 2200), "z": (10000, 13000)},
    "de_ancient": {"x": (-2900, 2200), "y": (-2900, 2200), "z": (-500, 1500)},
    "de_anubis": {"x": (-2800, 2500), "y": (-3300, 3000), "z": (-500, 1500)},
}

class CS2DeepLearningPipeline:
    """
    awpy kullanarak CS2 demo dosyalarını Derin Öğrenme modelleri için hazırlayan veri boru hattı.
    """
    def __init__(self, demo_path: str, samples_per_second: int = 2):
        self.demo_path = Path(demo_path)
        self.samples_per_second = samples_per_second
        self.demo: Optional[Demo] = None
        self.map_name: str = ""
        self.tickrate: int = 64
        self.bounds: Dict[str, tuple] = {}

    def parse_demo(self):
        """Demoyu okur ve gerekli tabloları yükler."""
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo dosyası bulunamadı: {self.demo_path}")

        logger.info(f"Demo ayrıştırılıyor: {self.demo_path}...")
        self.demo = Demo(str(self.demo_path))
        
        try:
            # Sadece ihtiyacımız olan sütunları çekmek performansı artırır
            # awpy.Demo.parse default olarak tüm veriyi çeker
            self.demo.parse() 
        except Exception as e:
            logger.error(f"Demo parse edilirken hata oluştu: {e}")
            raise

        self.map_name = self.demo.header.get("map_name", "unknown")
        self.tickrate = self.demo.tickrate or 64
        logger.info(f"Harita: {self.map_name} | Tickrate: {self.tickrate}")

        # Sınırları belirle
        if self.map_name in MAP_BOUNDS:
            self.bounds = MAP_BOUNDS[self.map_name]
            logger.info(f"Bilinen harita sınırları kullanılıyor: {self.map_name}")
        else:
            logger.warning(f"Harita '{self.map_name}' sınırları bilinmiyor! Veriden hesaplanıyor...")
            self.bounds = self._calculate_bounds_from_data()

    def _calculate_bounds_from_data(self) -> Dict[str, tuple]:
        """Tüm demo verisinden min/max koordinatları hesaplar."""
        df = self.demo.ticks
        if df is None or df.is_empty():
            return {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        
        return {
            "x": (float(df["X"].min()), float(df["X"].max())),
            "y": (float(df["Y"].min()), float(df["Y"].max())),
            "z": (float(df["Z"].min()), float(df["Z"].max()))
        }

    def _normalize(self, val: float, min_val: float, max_val: float) -> float:
        """Değeri 0-1 arasına normalize eder."""
        if max_val == min_val:
            return 0.5
        return float(np.clip((val - min_val) / (max_val - min_val), 0, 1))

    def process_to_dl_format(self) -> List[Dict[str, Any]]:
        """Veriyi derin öğrenme zaman serisi formatına dönüştürür."""
        if self.demo is None or self.demo.ticks is None:
            return []

        ticks_df = self.demo.ticks
        
        # 1. GEREKSİZ VERİLERİ FİLTRELE
        # Warmup ve Freezetime'ı çıkarıyoruz
        filter_cols = [c for c in ["is_warmup", "is_freezetime"] if c in ticks_df.columns]
        for col in filter_cols:
            ticks_df = ticks_df.filter(pl.col(col) == False)

        # 2. ÖRNEKLEME (DOWNSAMPLING)
        # Saniyelik pencere için kaç tick atlamalıyız?
        interval = max(1, int(self.tickrate / self.samples_per_second))
        
        # Round bazlı gruplama
        round_groups = ticks_df.group_by("round_num")
        
        all_rounds_data = []

        for round_info, round_ticks in round_groups:
            round_num = round_info[0] if isinstance(round_info, tuple) else round_info
            
            # Round içindeki benzersiz tick'leri al ve örnekle
            unique_ticks = round_ticks.select("tick").unique(maintain_order=True)["tick"].to_list()
            sampled_ticks = unique_ticks[::interval]
            
            sequence_data = []
            
            for tick in sampled_ticks:
                tick_subset = round_ticks.filter(pl.col("tick") == tick)
                if tick_subset.is_empty():
                    continue
                
                # Oyuncu verilerini topla (Sabit sıra için SteamID'ye göre sıralıyoruz)
                tick_subset = tick_subset.sort("steamid")
                
                players_features = []
                last_p = {}
                for p in tick_subset.to_dicts():
                    last_p = p
                    # TAKTİKSEL ÖZELLİKLER VE NORMALİZASYON
                    player_entry = {
                        "side": 1 if p.get("team_name") == "CT" else 0,
                        "hp": int(p.get("health", 0)) / 100.0,
                        "is_alive": 1 if p.get("is_alive", False) else 0,
                        "x": self._normalize(p.get("X", 0), *self.bounds["x"]),
                        "y": self._normalize(p.get("Y", 0), *self.bounds["y"]),
                        "z": self._normalize(p.get("Z", 0), *self.bounds["z"]),
                        "yaw": (float(p.get("yaw", 0)) + 180) / 360.0,
                        "pitch": (float(p.get("pitch", 0)) + 90) / 180.0
                    }
                    players_features.append(player_entry)

                # RAUND DURUM MATRİSİ ÖGESİ
                state_step = {
                    "tick": int(tick),
                    "round_time_remaining": float(last_p.get("time_remaining", 0)) if "time_remaining" in last_p else None,
                    "players": players_features
                }
                sequence_data.append(state_step)

            if sequence_data:
                all_rounds_data.append({
                    "round_num": int(round_num),
                    "sequence": sequence_data
                })

        # Raundları numarasina göre sirala
        all_rounds_data.sort(key=lambda x: x["round_num"])
        return all_rounds_data

def main():
    parser = argparse.ArgumentParser(description="CS2 Demo Deep Learning Data Pipeline")
    parser.add_argument("input", help="Ayrıştırılacak .dem dosyası")
    parser.add_argument("--output", "-o", help="Çıktı JSON dosyası (varsayılan: <demo_name>_dl.json)")
    parser.add_argument("--fps", type=int, default=1, help="Saniyede kaç örneklem alınacak (1 veya 2 önerilir)")
    
    args = parser.parse_args()
    
    demo_path = Path(args.input)
    output_path = args.output or demo_path.with_name(f"{demo_path.stem}_dl.json")
    
    pipeline = CS2DeepLearningPipeline(str(demo_path), samples_per_second=args.fps)
    
    try:
        pipeline.parse_demo()
        data = pipeline.process_to_dl_format()
        
        final_output = {
            "metadata": {
                "demo": demo_path.name,
                "map": pipeline.map_name,
                "sampling_fps": args.fps,
                "normalization_bounds": pipeline.bounds
            },
            "rounds": data
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Başarılı! Veri kaydedildi: {output_path}")
        logger.info(f"Toplam Raund: {len(data)}")
        
    except Exception as e:
        logger.error(f"HATA: {str(e)}")

if __name__ == "__main__":
    main()
