#!/bin/bash


# Path to the test list file
UT_LIST_FILE="all_test_file_names.txt"
UT_LOG_FOLDER="ut_logs"
mkdir -p "$UT_LOG_FOLDER"

find . -type f -name "test*.py" | sed 's#\./##g' | sed 's#\.py#\.py 1800#g'  |& tee "$UT_LIST_FILE" 

# Check if the test list file exists
if [ ! -f "$UT_LIST_FILE" ]; then
  echo "File not found: $UT_LIST_FILE"
  exit 1
fi

# Read the file line by line
while IFS= read -r line || [ -n "$line" ]; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" == \#* ]] && continue

  # Extract test file and timeout from the line
  TEST_FILE=$(echo "$line" | awk '{print $1}')
  TIMEOUT=$(echo "$line" | awk '{print $2}')

  # Validate that both fields are present
  if [ -z "$TEST_FILE" ] || [ -z "$TIMEOUT" ]; then
    echo "Invalid format: $line"
    continue
  fi

  echo ""
  echo "Running: $TEST_FILE with timeout ${TIMEOUT}s"

  # Run pytest with timeout
  timeout "${TIMEOUT}"s pytest "$TEST_FILE" |& tee "${UT_LOG_FOLDER}/${TEST_FILE}.log"
  STATUS=$?

  # Interpret exit status
  if [ $STATUS -eq 124 ]; then
    echo "Timeout: $TEST_FILE exceeded ${TIMEOUT}s"
  elif [ $STATUS -ne 0 ]; then
    echo "Failed: $TEST_FILE (exit code: $STATUS)"
  else
    echo "Passed: $TEST_FILE"
  fi

done < "$UT_LIST_FILE" 