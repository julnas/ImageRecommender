from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from imagehash import ImageHash


class Recommender:
    def __init__(
        self,
        db,
        loader,
        metrics: Dict[str, object],
    ):
        self.db = db
        self.loader = loader
        self.metrics = metrics

    # -------------------------------------------------------------------------
    # Feature combination
    # -------------------------------------------------------------------------

    @staticmethod
    def combine_color_features(
        feature_list,
        weights=None,
    ):
        """
        Combine multiple color features using weighted averaging.

        Each feature is expected to have the format:

            ((r, g, b), (h, s, l))

        Each channel is combined independently and L1-normalized.

        Returns:
            ((r, g, b), (h, s, l))
        """

        if not feature_list:
            raise ValueError(
                "combine_color_features: feature_list is empty."
            )

        # Validate the expected feature structure.
        try:
            (r0, g0, b0), (h0, s0, l0) = feature_list[0]
        except Exception:
            raise ValueError(
                "Expected each color feature to have the format "
                "((r, g, b), (h, s, l))."
            )

        # Convert arrays to flattened floating-point arrays.
        def as1d(x):
            return np.asarray(
                x,
                dtype=float,
            ).ravel()

        rgb_lists = list(
            zip(
                *[
                    tuple(
                        map(
                            as1d,
                            feature[0],
                        )
                    )
                    for feature in feature_list
                ]
            )
        )

        hsl_lists = list(
            zip(
                *[
                    tuple(
                        map(
                            as1d,
                            feature[1],
                        )
                    )
                    for feature in feature_list
                ]
            )
        )

        # Make sure all channels have the same size.
        channel_names = [
            "R",
            "G",
            "B",
            "H",
            "S",
            "L",
        ]

        for channel_list, name in zip(
            rgb_lists + hsl_lists,
            channel_names,
        ):
            lengths = {
                arr.size
                for arr in channel_list
            }

            if len(lengths) != 1:
                raise ValueError(
                    f"Channel {name} has inconsistent lengths: "
                    f"{sorted(lengths)}"
                )

        n = len(feature_list)

        # Use equal weights by default.
        if weights is None:
            weights = np.ones(
                n,
                dtype=float,
            ) / n

        else:
            weights = np.asarray(
                weights,
                dtype=float,
            )

            if weights.size != n:
                raise ValueError(
                    "Number of weights must match "
                    "the number of features."
                )

            weight_sum = weights.sum()

            if weight_sum <= 0:
                raise ValueError(
                    "Sum of weights must be greater than zero."
                )

            weights = weights / weight_sum

        def mix_channel(arrays):
            combined = np.zeros_like(
                arrays[0],
                dtype=float,
            )

            for weight, array in zip(
                weights,
                arrays,
            ):
                combined += weight * array

            total = combined.sum()

            if total > 0:
                combined /= total

            return combined

        r = mix_channel(
            rgb_lists[0]
        )

        g = mix_channel(
            rgb_lists[1]
        )

        b = mix_channel(
            rgb_lists[2]
        )

        h = mix_channel(
            hsl_lists[0]
        )

        s = mix_channel(
            hsl_lists[1]
        )

        l = mix_channel(
            hsl_lists[2]
        )

        return (
            r,
            g,
            b,
        ), (
            h,
            s,
            l,
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def combine_embeddings(
        embed_list: List[np.ndarray],
        weights: List[float] = None,
    ) -> np.ndarray:
        """
        Combine multiple embeddings using weighted averaging.

        Each embedding is normalized before combining.
        The resulting embedding is normalized again afterwards.
        """

        if not embed_list:
            raise ValueError(
                "combine_embeddings: embed_list is empty."
            )

        embed_list = [
            np.asarray(
                embedding,
                dtype=float,
            )
            for embedding in embed_list
        ]

        shape = embed_list[0].shape

        if any(
            embedding.shape != shape
            for embedding in embed_list
        ):
            raise ValueError(
                "All embeddings must have the same shape."
            )

        if weights is None:
            weights = (
                np.ones(
                    len(embed_list),
                    dtype=float,
                )
                / len(embed_list)
            )

        else:
            weights = np.asarray(
                weights,
                dtype=float,
            )

            if weights.size != len(embed_list):
                raise ValueError(
                    "Number of weights must match "
                    "the number of embeddings."
                )

            weight_sum = weights.sum()

            if weight_sum <= 0:
                raise ValueError(
                    "Sum of weights must be greater than zero."
                )

            weights = weights / weight_sum

        # Normalize each embedding first.
        normalized = [
            embedding
            / (
                np.linalg.norm(embedding)
                + 1e-12
            )
            for embedding in embed_list
        ]

        combined = np.zeros_like(
            normalized[0],
            dtype=float,
        )

        for embedding, weight in zip(
            normalized,
            weights,
        ):
            combined += weight * embedding

        # Normalize the combined embedding.
        combined /= (
            np.linalg.norm(combined)
            + 1e-12
        )

        return combined

    # -------------------------------------------------------------------------

    @staticmethod
    def combine_hashes_majority(
        hash_list: List[
            Union[
                ImageHash,
                np.ndarray,
                List[int],
                List[bool],
            ]
        ],
    ) -> ImageHash:
        """
        Combine multiple perceptual hashes using bitwise majority voting.

        Supported input types:
            - imagehash.ImageHash
            - NumPy arrays
            - lists/tuples containing bits

        Returns:
            An ImageHash that can be passed directly to
            HashingSimilarity.find_similar().
        """

        if not hash_list:
            raise ValueError(
                "combine_hashes_majority: hash_list is empty."
            )

        def to_bits(value):
            # imagehash.ImageHash
            if isinstance(
                value,
                ImageHash,
            ):
                return np.asarray(
                    value.hash,
                    dtype=np.uint8,
                ).ravel()

            # NumPy arrays / lists / tuples
            if isinstance(
                value,
                (
                    np.ndarray,
                    list,
                    tuple,
                ),
            ):
                array = np.asarray(
                    value,
                    dtype=np.uint8,
                ).ravel()

                # Convert values to binary.
                return (
                    array != 0
                ).astype(
                    np.uint8
                )

            raise TypeError(
                "Unsupported hash type: "
                f"{type(value).__name__}"
            )

        # Convert all hashes to flat bit arrays.
        bit_arrays = [
            to_bits(hash_value)
            for hash_value in hash_list
        ]

        # All hashes must have the same number of bits.
        bit_lengths = {
            array.size
            for array in bit_arrays
        }

        if len(bit_lengths) != 1:
            raise ValueError(
                "All hashes must have the same number "
                f"of bits. Got: {sorted(bit_lengths)}"
            )

        # Keep the original ImageHash shape.
        first_hash = hash_list[0]

        if isinstance(
            first_hash,
            ImageHash,
        ):
            hash_shape = first_hash.hash.shape

        else:
            hash_shape = (
                bit_arrays[0].shape
            )

        # Shape:
        #   number of hashes x number of bits
        array_2d = np.vstack(
            bit_arrays
        )

        # Count the number of votes for bit = 1.
        votes = array_2d.sum(
            axis=0
        )

        # Majority vote.
        majority = (
            votes
            >= (
                array_2d.shape[0]
                / 2
            )
        )

        # Restore the original hash shape.
        majority = majority.reshape(
            hash_shape
        )

        return ImageHash(
            majority
        )

    # -------------------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------------------

    def recommend(
        self,
        input_image,
        best_k: int = 1,
    ) -> Dict[str, list]:
        """
        Return the top-k similar image IDs for each configured metric.

        If input_image is a list, the features are calculated separately
        for every image and then combined according to the metric type.

        Supported metric types:
            - color
            - embedding
            - hash
            - hashing
            - phash
            - dhash
            - ahash
        """

        if best_k <= 0:
            raise ValueError(
                "best_k must be greater than zero."
            )

        results: Dict[str, list] = {}

        for metric_name, metric in self.metrics.items():

            # -------------------------------------------------------------
            # Multiple input images
            # -------------------------------------------------------------

            if isinstance(
                input_image,
                list,
            ):

                if len(input_image) == 0:
                    raise ValueError(
                        "recommend: input_image is an empty list."
                    )

                # Calculate one feature per input image.
                feature_list = [
                    metric.compute_feature(
                        image
                    )
                    for image in input_image
                ]

                # Combine features according to the metric.
                if metric_name == "color":

                    query_vector = (
                        Recommender.combine_color_features(
                            feature_list
                        )
                    )

                elif metric_name == "embedding":

                    query_vector = (
                        Recommender.combine_embeddings(
                            feature_list
                        )
                    )

                elif metric_name in (
                    "hash",
                    "hashing",
                    "phash",
                    "dhash",
                    "ahash",
                ):

                    query_vector = (
                        Recommender.combine_hashes_majority(
                            feature_list
                        )
                    )

                else:
                    raise ValueError(
                        "Unknown metric for list fusion: "
                        f"{metric_name}"
                    )

            # -------------------------------------------------------------
            # Single input image
            # -------------------------------------------------------------

            else:

                query_vector = (
                    metric.compute_feature(
                        input_image
                    )
                )

            # -------------------------------------------------------------
            # Perform similarity search.
            #
            # HashingSimilarity now handles the compact 8-byte
            # representation internally. The recommender still works
            # with ImageHash objects as query features.
            # -------------------------------------------------------------

            similar_ids = metric.find_similar(
                query_vector,
                best_k=best_k,
            )

            results[metric_name] = similar_ids

        return results            if len(lens) != 1:
                raise ValueError(f"Kanal {name} hat unterschiedliche Längen: {sorted(lens)}")

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
            if s > 0: c /= s  # L1 pro Kanal
            return c

        r = mix_channel(rgb_lists[0]); g = mix_channel(rgb_lists[1]); b = mix_channel(rgb_lists[2])
        h = mix_channel(hsl_lists[0]); s = mix_channel(hsl_lists[1]); l = mix_channel(hsl_lists[2])

        return (r, g, b), (h, s, l)

    @staticmethod
    def combine_embeddings(embed_list: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
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
        combined /= (np.linalg.norm(combined) + 1e-12)
        return combined

    @staticmethod
    def combine_hashes_majority(hash_list: List[Union[np.ndarray, List[int], List[bool]]]) -> np.ndarray:
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
                arr = np.fromiter((1 if ch == "1" else 0 for ch in bstr), dtype=int, count=n_bits)
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

        arr2d = np.vstack(padded)               # shape: (n_hashes, n_bits)
        votes = arr2d.sum(axis=0)               # Stimmen für 1 je Bit
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
                    raise ValueError(f"Unbekannte Metrik für Listenfusion: {metric_name}")
            else:
                # Einzelbild
                query_vector = metric.compute_feature(input_image)

            similar_ids = metric.find_similar(query_vector, best_k=best_k)
            results[metric_name] = similar_ids

        return results
