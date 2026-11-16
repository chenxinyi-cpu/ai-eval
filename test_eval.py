import evaluate

metric = evaluate.load("accuracy")
result = metric.compute(predictions=[0,1,1],references=[0,1,0])
print(result)