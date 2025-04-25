from typing import Dict

class Recommender:
    def __init__(self, db, loader, metrics: Dict[str, object]):
        """
        coordinates loading the input image, 
        aplying similarity measures, and returning results.

        Parameters:
        db: instance of DatabaseManager
        loader: instance of ImageLoader
        metrics: dictionary of similarity metric instances 
        """
        self.db = db
        self.loader = loader
        self.metrics = metrics  

    def recommend(self, input_image_id: int, best_k: int = 1) -> Dict[str, list]:
        """
        Finds the best similar image for a given input image ID across all specified similarity metrics.
        u can change the best k to any number u want. right now its 1, cuz we get an output per metric.

        Parameters:
        input_image_id: ID of the input image
        best_k: number of top similar images to return per metrik (so later we should change the code, so the metrics will be combined and we get the best 5 images overall)

        Returns:
        A list of top-k image IDs
        """
        
        #oad input image
        input_image = self.loader.load_image(input_image_id)

        results = {}

        #for each metric, compute similarity and retrieve best-k matches
        for metric_name, metric in self.metrics.items():
            query_vector = metric.compute_feature(input_image)  #compute feature vector for the input image
            similar_image_ids = metric.find_similar(query_vector, best_k=best_k)    #search for similar images
            results[metric_name] = similar_image_ids        #safe the results

        return results
