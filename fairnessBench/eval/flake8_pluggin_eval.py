import ast
import sys 
if sys.version_info < (3, 8):
    import importlib_metadata
else: 
    import importlib.metadata as importlib_metadata
class Fairnessevaluator:
    name = __name__
    version = importlib_metadata.version(__name__)

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        self.score = 0
        self.issues = []
    def run(self): 
        self.check_data_collection()
        # self.check_missing_value_handling
        self.check_categorical_encoding()
        self.check_bias_mitigation()
        self.check_fairness_metrics()
        self.check_model_training()
        self.check_evaluation()

        for line, col, msg in self.issues:
            yield line, col, msg, type(self)
        print(f"Fairness Score: {self.score}/80")
    # format on how the error message should look like, it takes as input the line, column and the message   
    def add_issue(self, node, message, deduction=0):
        lineno = getattr(node, 'lineno', 1)
        col_offset = getattr(node, 'col_offset', 0)
        self.issues.append((lineno, col_offset, message))
        self.score -= deduction    

    def check_data_collection(self):
        libs = ["pandas", "numpy", "sklearn", "datasets"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in libs and a.name not in found:
                        found.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module.split(".",1)[0] if node.module else ""
                if mod in libs and mod not in found:
                    found.append(mod)

        missing = [l for l in libs if l not in found]
        weight = 15
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA101: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            # no items , no +score, just message
            self.add_issue(anchor,
                "FNA101: No dataset processing library found (e.g., pandas, numpy, sklearn, datasets)"
            )

    def check_missing_value_handling(self):
        funcs = ["dropna", "fillna"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr in funcs:
                if node.attr not in found: found.append(node.attr)
            elif isinstance(node, ast.Name) and node.id in funcs:
                if node.id not in found: found.append(node.id)

        missing = [f for f in funcs if f not in found]
        weight = 10
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA102: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA102: No handling of missing values detected (e.g., dropna, fillna)"
            )

    def check_categorical_encoding(self):
        encs = ["get_dummies", "OneHotEncoder", "LabelEncoder"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and node.attr in encs:
                if node.attr not in found: found.append(node.attr)
            elif isinstance(node, ast.Name) and node.id in encs:
                if node.id not in found: found.append(node.id)

        missing = [e for e in encs if e not in found]
        weight = 10
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA103: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA103: No categorical encoding found (e.g., get_dummies, OneHotEncoder, LabelEncoder)"
                
            )
        
    def check_bias_mitigation(self):
        libs = ["aif360", "fairlearn", "equitas", "fairness_indicator"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".",1)[0] in libs and a.name.split(".",1)[0] not in found:
                        found.append(a.name.split(".",1)[0])
            elif isinstance(node, ast.ImportFrom):
                mod = node.module.split(".",1)[0] if node.module else ""
                if mod in libs and mod not in found:
                    found.append(mod)

        missing = [l for l in libs if l not in found]
        weight = 15
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA104: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA104: No bias mitigation techniques found (e.g., aif360, fairlearn, equitas, fairness_indicator)"
            )

    def check_fairness_metrics(self):
        mets = ["equalized_odds", "demographic_parity", "statistical_parity", "disparate_impact_ratio", "accuracy","average_abs_odds_difference", "average_odds_difference", "consistency","false_discovery_rate"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in mets:
                if node.name not in found: found.append(node.name)
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name in mets and name not in found:
                    found.append(name)

        missing = [m for m in mets if m not in found]
        weight = 10
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA105: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA105: No fairness metrics function found (e.g., equalized_odds, demographic_parity, statistical_parity, disparate_impact_ratio)"
                           
            )
         
    def check_model_training(self):
        terms = ["adversarial", "reweighting","DisparateImpactRemover","AdversarialDebiasing","ARTClassifier","PrejudiceRemover", "EqOddsPostprocessing","DeterministicReranking","GerryFairClassifier"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and any(t in node.name for t in terms):
                for t in terms:
                    if t in node.name and t not in found: found.append(t)
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name in terms and name not in found:
                    found.append(name)

        missing = [t for t in terms if t not in found]
        weight = 10
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA106: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA106: No fairness-aware training techniques found (e.g., adversarial, reweighting,DisparateImpactRemover,AdversarialDebiasing,ARTClassifier,PrejudiceRemover, EqOddsPostprocessing,DeterministicReranking,GerryFairClassifier)"
                
            )

    def check_evaluation(self):
        funcs = ["audit_bias", "disparate_impact_ratio"]
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name in funcs:
                if node.name not in found: found.append(node.name)
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name in funcs and name not in found:
                    found.append(name)

        missing = [f for f in funcs if f not in found]
        weight = 10
        anchor = next(ast.walk(self.tree))

        if found:
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA107: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA107: No fairness evaluation or auditing function found (e.g., audit_bias, disparate_impact_ratio)"
                
            )