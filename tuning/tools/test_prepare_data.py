import unittest

from tools.prepare_data import detect_duplicates

class TestDetectDuplicates(unittest.TestCase):

    def test_no_duplicates(self):
        """
        Each split has one row with unique test_id, comment, and answer.
        No duplicates (intra- or inter-split) should be detected.
        """
        dataset = {
            "train": [{
                "test_id": "1",
                "comment": "Unique comment train",
                "answers": [{"text": "Answer A", "answer": "А"}]
            }],
            "val": [{
                "test_id": "2",
                "comment": "Unique comment val",
                "answers": [{"text": "Answer B", "answer": "Б"}]
            }],
            "test": [{
                "test_id": "3",
                "comment": "Unique comment test",
                "answers": [{"text": "Answer C", "answer": "В"}]
            }]
        }
        dups_stats = detect_duplicates(dataset)
        # No duplicates expected
        self.assertEqual(dict(dups_stats), {})

    def test_intra_split_duplicates(self):
        """
        Two rows within the same split ("train") have identical comments and answers.
        This should trigger the intra-split duplicate counters for comment and answer.
        """
        duplicate_row = {
            "test_id": "10",
            "comment": "Same Comment",
            "answers": [{"text": "Duplicate Answer"}]
        }
        dataset = {
            "train": [duplicate_row, duplicate_row],
            "val": [],
            "test": []
        }
        dups_stats = detect_duplicates(dataset)
        expected = {
            "train_comment_in_set": 1,  # second occurrence triggers duplicate comment
            "train_answer_in_set": 1    # second occurrence triggers duplicate answer
        }
        self.assertEqual(dict(dups_stats), expected)

    def test_inter_split_duplicates(self):
        """
        A row appearing in "train" and a row appearing in "val" share the same test_id,
        comment, and answer. This should trigger inter-split duplicate counts when processing
        the row in the later split ("val"). Also, a unique row in "test" should not trigger duplicates.
        """
        # Row in "train"
        row_train = {
            "test_id": "dup",  # common id for inter-split duplicate
            "comment": "Duplicate Comment",
            "answers": [{"text": "Dup Answer"}]
        }
        # Row in "val" with identical values as row_train.
        row_val = {
            "test_id": "dup",  # same test_id as row_train
            "comment": "Duplicate Comment",  # identical (case-insensitive)
            "answers": [{"text": "Dup Answer"}]
        }
        # A unique row in "test"
        row_test = {
            "test_id": "unique",
            "comment": "Unique Test Comment",
            "answers": [{"text": "Unique Answer"}]
        }
        dataset = {
            "train": [row_train],
            "val": [row_val],
            "test": [row_test]
        }
        dups_stats = detect_duplicates(dataset)
        # Explanation:
        # When processing row_train in "train":
        #   - id_splits["train"] becomes {"dup"}
        #   - comments_set["train"] gets "duplicate comment"
        #   - answers_set["train"] gets "dup answer"
        #   - In inner loop over _split ("val", "test"):
        #       * For "val": Since id_splits["val"] is still empty, no inter-split duplicates occur.
        #       * For "test": id_splits["test"] is empty, so nothing.
        #
        # When processing row_val in "val":
        #   - id_splits["val"] becomes {"dup"}
        #   - comments_set["val"] gets "duplicate comment"
        #   - answers_set["val"] gets "dup answer"
        #   - In inner loop over _split:
        #       * For _split "train": 
        #             "dup" is in id_splits["train"] → triggers "val_id_in_train" += 1.
        #             "duplicate comment" is in comments_set["train"] → triggers "val_comment_in_train" += 1.
        #             "dup answer" is in answers_set["train"] → triggers "val_answer_in_train" += 1.
        #       * For _split "test": id_splits["test"] is empty → no duplicates.
        expected = {
            "val_id_in_train": 1,
            "val_comment_in_train": 1,
            "val_answer_in_train": 1
        }
        self.assertEqual(dict(dups_stats), expected)

    def test_complex_duplicates(self):
        """
        A complex scenario with multiple rows in each split.
          - "train" has three rows:
              * Two rows with test_id "1", comment "A", answer "X" (duplicates)
              * One row with test_id "2", comment "B", answer "Y" (unique)
          - "val" has two rows:
              * One row with test_id "1", comment "A", answer "X" (duplicate of train)
              * One row with test_id "3", comment "C", answer "Z" (unique)
          - "test" has two rows:
              * One row with test_id "2", comment "B", answer "Y" (duplicate of train)
              * One row with test_id "4", comment "D", answer "W" (unique)
        We then verify the duplicate counts across intra-split and inter-split comparisons.
        """
        dataset = {
            "train": [
                {"test_id": "1", "comment": "A", "answers": [{"text": "X"}]},
                {"test_id": "1", "comment": "A", "answers": [{"text": "X"}]},  # duplicate within train
                {"test_id": "2", "comment": "B", "answers": [{"text": "Y"}]}
            ],
            "val": [
                {"test_id": "1", "comment": "A", "answers": [{"text": "X"}]},  # duplicate across splits (train/val)
                {"test_id": "3", "comment": "C", "answers": [{"text": "Z"}]}
            ],
            "test": [
                {"test_id": "2", "comment": "B", "answers": [{"text": "Y"}]},  # duplicate across splits (train/test)
                {"test_id": "4", "comment": "D", "answers": [{"text": "W"}]}
            ]
        }
        dups_stats = detect_duplicates(dataset)
        # Expected explanation:
        # Processing "train":
        #   - Duplicate within train: "train_comment_in_set" = 1, "train_answer_in_set" = 1.
        # Processing "val" (row with test_id "1", "A", "X"):
        #   - "1" is in id_splits["train"], "a" is in comments_set["train"], and "x" is in answers_set["train"],
        #     so for that row: "val_id_in_train", "val_comment_in_train", and "val_answer_in_train" each increment by 1.
        # Processing "test" (row with test_id "2", "B", "Y"):
        #   - "2" is in id_splits["train"], "b" is in comments_set["train"], and "y" is in answers_set["train"],
        #     so for that row: "test_id_in_train", "test_comment_in_train", and "test_answer_in_train" each increment by 1.
        expected = {
            "train_comment_in_set": 1,
            "train_answer_in_set": 1,
            "val_id_in_train": 1,
            "val_comment_in_train": 1,
            "val_answer_in_train": 1,
            "test_id_in_train": 1,
            "test_comment_in_train": 1,
            "test_answer_in_train": 1
        }
        self.assertEqual(dict(dups_stats), expected)

    def test_all_duplicates(self):
        """
        Every row in every split is identical.
        Each split has two identical rows with:
            test_id: "dup", comment: "dup", answer: "dup"
        This should trigger:
          - Intra-split duplicates: one duplicate comment and one duplicate answer per split.
          - Inter-split duplicates:
              * For rows in "val": each row will detect duplicates in "train".
              * For rows in "test": each row will detect duplicates in "train" and "val".
        Since sets are used, each split's id, comment, and answer sets only contain "dup".
        """
        row = {"test_id": "dup", "comment": "dup", "answers": [{"text": "dup"}]}
        dataset = {
            "train": [row, row],
            "val": [row, row],
            "test": [row, row]
        }
        dups_stats = detect_duplicates(dataset)
        expected = {
            # Intra-split duplicates:
            "train_comment_in_set": 1,
            "train_answer_in_set": 1,
            "val_comment_in_set": 1,
            "val_answer_in_set": 1,
            "test_comment_in_set": 1,
            "test_answer_in_set": 1,
            # Inter-split duplicates:
            # When processing "val": each row sees that "dup" exists in "train"
            "val_id_in_train": 2,
            "val_comment_in_train": 2,
            "val_answer_in_train": 2,
            # When processing "test": each row sees that "dup" exists in both "train" and "val"
            "test_id_in_train": 2,
            "test_comment_in_train": 2,
            "test_answer_in_train": 2,
            "test_id_in_val": 2,
            "test_comment_in_val": 2,
            "test_answer_in_val": 2
        }
        self.assertEqual(dict(dups_stats), expected)


if __name__ == '__main__':
    unittest.main()