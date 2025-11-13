Task 1: AI-Powered Code Completion AnalysisCode SnippetsHere is a comparison of a manually written Python function and an AI-suggested (GitHub Copilot) alternative to sort a list of dictionaries.Prompt given to AI:# python function to sort a list of dictionaries by a specific key
def sort_dicts_by_key_ai(data, key):
Version 1: Manual Implementationdef sort_dicts_by_key_manual(data, key):
  """
  Sorts a list of dictionaries using a lambda function.
  """
  return sorted(data, key=lambda x: x[key])

## --- Example Usage ---
## employees = [
##     {'name': 'John', 'age': 30},
##     {'name': 'Jane', 'age': 25},
## ]
## sorted_employees = sort_dicts_by_key_manual(employees, 'age')
## print(sorted_employees)
Version 2: AI-Suggested Implementation (GitHub Copilot)from operator import itemgetter

def sort_dicts_by_key_ai(data, key):
  """
  Sorts a list of dictionaries using operator.itemgetter.
  Suggested by GitHub Copilot.
  """
  return sorted(data, key=itemgetter(key))

## --- Example Usage ---
## employees = [
##     {'name': 'John', 'age': 30},
##     {'name': 'Jane', 'age': 25},
## ]
## sorted_employees = sort_dicts_by_key_ai(employees, 'age')
## print(sorted_employees)

# Analysis 
The AI-suggested code is not only correct but more efficient than the standard manual implementation. My manual version used a lambda function, which is a common and perfectly readable Pythonic solution. The lambda function is defined at runtime and then called for every element in the list during the sort.In contrast, GitHub Copilot suggested using from operator import itemgetter. The itemgetter(key) function is a C-optimized callable that accesses the key from each dictionary. Because it is implemented in C and avoids the overhead of a Python-level lambda function call for each item, itemgetter is demonstrably faster.For a small list, the difference is negligible. However, for a list with millions of dictionaries, the AI's suggestion provides a significant performance improvement. This is a prime example of an AI tool reducing development time not just by writing code, but by suggesting a more optimal and high-performance solution than a developer might have written by default. The AI-generated code is cleaner, faster, and requires no additional thought.