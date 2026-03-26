import unittest
from sklearn.metrics import classification_report, accuracy_score
from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer

class TestModelPerformance(unittest.TestCase):
    def setUp(self):
        self.analyzer = MoodAnalyzer()
        self.texts = SAMPLE_POSTS
        self.labels = TRUE_LABELS

    def test_accuracy_and_recall(self):
        """
        Runs the model on the dataset and prints accuracy and recall metrics.
        """
        print("\n\n" + "="*40)
        print("Model Performance Analysis")
        print("="*40)

        # distinct labels for classification_report
        target_names = sorted(list(set(self.labels)))

        # Get predictions
        predictions = [self.analyzer.predict_label(text) for text in self.texts]

        # Calculate overall accuracy
        accuracy = accuracy_score(self.labels, predictions)
        print(f"\nOverall Accuracy: {accuracy:.2%}")

        # detailed report
        print("\nDetailed Classification Report (includes Recall):")
        if len(self.labels) > 0:
            # check if sklearn is available, otherwise this might fail if not installed
            # but user has it in requirements.txt
            report = classification_report(
                self.labels, 
                predictions, 
                zero_division=0
            ) 
            print(report)
        else:
            print("No labels to analyze.")
        
        print("="*40 + "\n")

        # Optional: assert some baseline performance
        # self.assertGreaterEqual(accuracy, 0.5, "Accuracy should be at least 50%")

if __name__ == '__main__':
    unittest.main()
