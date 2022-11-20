#!/bin/bash
echo "model name"
read clf_name
echo "state number"
read random_state

CMD ["python3", "./mlops/plot_graphs.py" "--clf_name" $clf_name "--random_state" $random_state]]
