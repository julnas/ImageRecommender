from typing import Dict, List, Union
import numpy as np
from imagehash import ImageHash


class Recommender:
    def __init__(self, db, loader, metrics: Dict[str, object]):
        self.db = db
        self.loader = loader
        self.metrics = metrics

    # ---- Kombinationsfunktionen -------------------------------------------
    @staticmethod
    def combine_color_features(feature_list, weights=None):
        """
        Kombiniert mehrere Color-Features (jeweils ((r,g,b), (h,s,l))) zu einem einzigen
        durch gewichtetes Addieren pro Kanal + L1-Normalisierung pro Kanal.
        Rückgabeformat: ((r,g,b), (h,s,l)) — kompatibel zu ColorSimilarity.find_similar.
        """
        if not feature_list:
            raise ValueError("combine_color_features: feature_list ist leer.")

        # Unpack prüfen
        try:
            (r0, g0, b0), (h0, s0, l0) = feature_list[0]
        except Exception:
            raise ValueError("Erwarte für jedes Feature das Format ((r,g,b),(h,s,l)).")

        # Alle zu float + 1D
        def as1d(x):
            return np.asarray(x, dtype=float).ravel()

        rgb_lists = list(
            zip(*[tuple(map(as1d, f[0])) for f in feature_list])
        )  # -> [ [r_i], [g_i], [b_i] ]
        hsl_lists = list(
            zip(*[tuple(map(as1d, f[1])) for f in feature_list])
        )  # -> [ [h_i], [s_i], [l_i] ]

        # Konsistenz: gleiche Länge pro Kanal
        for ch_list, name in zip(rgb_lists + hsl_lists, ["R", "G", "B", "H", "S", "L"]):
            lens = {arr.size for arr in ch_list}
            if len(lens) != 1:
                raise ValueError(
                    f"Kanal {name} hat unterschiedliche Längen: {sorted(lens)}"
                )

        n = len(feature_list)
        if weights is None:
            weights = np.ones(n, dtype=float) / n
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / (weights.sum() + 1e-12)

        def mix_channel(arrs):
            c = np.zeros_like(arrs[0], dtype=float)
            for w, a in zip(weights, arrs):
                c += w * a
            s = c.sum()
            if s > 0:
                c /= s  # L1 pro Kanal
            return c

        r = mix_channel(rgb_lists[0])
        g = mix_channel(rgb_lists[1])
        b = mix_channel(rgb_lists[2])
        h = mix_channel(hsl_lists[0])
        s = mix_channel(hsl_lists[1])
        l = mix_channel(hsl_lists[2])

        return (r, g, b), (h, s, l)

    @staticmethod
    def combine_embeddings(
        embed_list: List[np.ndarray], weights: List[float] = None
    ) -> np.ndarray:
        if not embed_list:
            raise ValueError("combine_embeddings: embed_list ist leer.")
        embed_list = [np.asarray(e, dtype=float) for e in embed_list]
        D = embed_list[0].shape
        if any(e.shape != D for e in embed_list):
            raise ValueError("Alle Embeddings müssen dieselbe Form haben.")
        if weights is None:
            weights = np.ones(len(embed_list), dtype=float) / len(embed_list)
        else:
            weights = np.asarray(weights, dtype=float)
            weights = weights / (weights.sum() + 1e-12)
        normed = [e / (np.linalg.norm(e) + 1e-12) for e in embed_list]
        combined = np.zeros_like(normed[0], dtype=float)
        for e, w in zip(normed, weights):
            combined += w * e
        combined /= np.linalg.norm(combined) + 1e-12
        return combined

    @staticmethod
    def combine_hashes_majority(
        hash_list: List[Union[np.ndarray, List[int], List[bool]]],
    ) -> np.ndarray:
        """
        Kombiniert mehrere Hashes per bitweiser Mehrheitsabstimmung.
        Akzeptiert: ImageHash-Objekte, Arrays/Listen (0/1/bool), Hex-Strings/Bytes.
        Rückgabe: np.ndarray der Länge n_bits (0/1).
        """
        if not hash_list:
            raise ValueError("combine_hashes_majority: hash_list ist leer.")

        def to_bits(x):
            import numpy as np

            # 1) imagehash.ImageHash
            if hasattr(x, "hash"):
                # x.hash ist ein bool-Array (H x W)
                arr = np.asarray(x.hash, dtype=int).ravel()
                return arr
            # 2) numpy/list/tuple
            if isinstance(x, (np.ndarray, list, tuple)):
                arr = np.asarray(x).astype(int).ravel()
                # auf 0/1 bringen, falls Werte nicht strikt binär sind
                arr = (arr != 0).astype(int)
                return arr
            # 3) Hex-String / Bytes
            if isinstance(x, (str, bytes)):
                hexstr = x.decode() if isinstance(x, bytes) else x
                hexstr = hexstr.strip().lower()
                if hexstr.startswith("0x"):
                    hexstr = hexstr[2:]
                if any(ch not in "0123456789abcdef" for ch in hexstr):
                    raise TypeError(f"Ungültiger Hash-String: {x!r}")
                n_bits = len(hexstr) * 4
                intval = int(hexstr, 16) if hexstr else 0
                bstr = bin(intval)[2:].zfill(n_bits)  # Binärstring mit führenden Nullen
                arr = np.fromiter(
                    (1 if ch == "1" else 0 for ch in bstr), dtype=int, count=n_bits
                )
                return arr
            raise TypeError(f"Nicht unterstützter Hash-Typ: {type(x).__name__}")

        # Alle in Bits wandeln
        bit_arrays = [to_bits(h) for h in hash_list]

        # Längen angleichen (links mit 0 auffüllen), weil Hex-Strings oft führende Nullen verlieren
        max_len = max(arr.size for arr in bit_arrays)
        padded = []
        for arr in bit_arrays:
            if arr.size < max_len:
                pad = np.zeros(max_len - arr.size, dtype=int)
                arr = np.concatenate([pad, arr], axis=0)  # links auffüllen
            padded.append(arr)

        arr2d = np.vstack(padded)  # shape: (n_hashes, n_bits)
        votes = arr2d.sum(axis=0)  # Stimmen für 1 je Bit
        majority = (votes >= (arr2d.shape[0] / 2)).astype(int)
        return ImageHash(majority)

    # ---- Empfehlung ---------------------------------------------------------
    def recommend(self, input_image, best_k: int = 1) -> Dict[str, list]:
        """
        Gibt je Metrik die Top-k ähnlichen Bild-IDs zurück.
        - Wenn input_image eine Liste ist, werden die Features je nach Metrik
          korrekt kombiniert (Histogramm, Embedding, Hash).
        """
        results: Dict[str, list] = {}

        for metric_name, metric in self.metrics.items():
            # Features berechnen
            if isinstance(input_image, list):
                if len(input_image) == 0:
                    raise ValueError("recommend: input_image ist eine leere Liste.")
                feature_list = [metric.compute_feature(img) for img in input_image]

                # Kombinieren je Modalität
                if metric_name == "color":
                    query_vector = Recommender.combine_color_features(feature_list)
                elif metric_name == "embedding":
                    query_vector = Recommender.combine_embeddings(feature_list)
                elif metric_name in ("hash", "hashing", "phash", "dhash", "ahash"):
                    query_vector = Recommender.combine_hashes_majority(feature_list)
                else:
                    raise ValueError(
                        f"Unbekannte Metrik für Listenfusion: {metric_name}"
                    )
            else:
                # Einzelbild
                query_vector = metric.compute_feature(input_image)

            similar_ids = metric.find_similar(query_vector, best_k=best_k)
            results[metric_name] = similar_ids

        return results
