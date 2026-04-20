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
        self.check_column_identification()
        self.check_preprocessing()
        self.check_categorical_encoding()
        self.check_sensitive_features()
        self.check_bias_mitigation_libraries()
        self.check_fairness_metrics()
        self.check_model_training()
        self.check_evaluation()

        for line, col, msg in self.issues:
            yield line, col, msg, type(self)
        print(f"Fairness Score: {self.score}")
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
        weight = 10
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


    def check_column_identification(self):
            
        found_numeric = False
        found_categorical = False

        for node in ast.walk(self.tree):
            # detect df.select_dtypes(include=[...])
            if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "select_dtypes"):
                for kw in node.keywords:
                    if kw.arg == "include" and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            if (isinstance(elt, ast.Constant)
                                and elt.value in ("int64", "float64")):
                                found_numeric = True
                            if (isinstance(elt, ast.Constant)
                                and elt.value == "object"):
                                found_categorical = True

        anchor = next(ast.walk(self.tree))
        weight = 10

        found = []
        if found_numeric:
            found.append("numeric")
        if found_categorical:
            found.append("categorical")

        missing = []
        if not found_numeric:
            missing.append("numeric")
        if not found_categorical:
            missing.append("categorical")

        if found:
            # award full weight
            self.score += weight
            fstr = ", ".join(found)
            mstr = ", ".join(missing)
            self.add_issue(anchor,
                f"FNA102: Found {fstr} column identification, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA102: No numeric or categorical column identification found "
                "(e.g., df.select_dtypes(include=['int64','float64']).columns and "
                "df.select_dtypes(include=['object']).columns)"
            )

    def check_preprocessing(self):
        pre = ["StandardScaler", "ColumnTransformer", "values.ravel()"]
        found = []
        for node in ast.walk(self.tree):
        # detect StandardScaler / ColumnTransformer as Name or Attribute
            if isinstance(node, ast.Name) and node.id in ("StandardScaler", "ColumnTransformer"):
                if node.id not in found:
                    found.append(node.id)
    
            elif isinstance(node, ast.Attribute):
                # e.g. ColumnTransformer(...)
                if node.attr in ("StandardScaler", "ColumnTransformer") and node.attr not in found:
                    found.append(node.attr)
                # detect values.ravel()
                elif node.attr == "ravel":
                    val = node.value
                    if isinstance(val, ast.Attribute) and val.attr == "values":
                        if "values.ravel()" not in found:
                            found.append("values.ravel()")

        missing = [e for e in pre if e not in found]
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
                "FNA103: No preprocessing encoding found (e.g., StandardScaler, ColumnTransformer, values.ravel())"
                
            )

    def check_categorical_encoding(self):
        encs = ["get_dummies", "OneHotEncoder", "LabelEncoder","OrdinalEncoder"]
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
                f"FNA104: Found {fstr}, but didn’t find {mstr}, +{weight}"
            )
        else:
            self.add_issue(anchor,
                "FNA104: No categorical encoding found (e.g., get_dummies, OneHotEncoder, LabelEncoder,OrdinalEncoder)"
                
            )
    def check_sensitive_features(self):
        
        """Check that sensitive/protected features are explicitly identified,
        which is a prerequisite for meaningful fairness analysis."""
        
        sensitive_keywords = [
            "sensitive_features", "sensitive_feature",
            "protected_attribute", "protected_attributes",
            "sensitive_attr", "sensitive_attrs",
            "protected_feature", "protected_features",
            "privileged_groups", "unprivileged_groups",
            "protected_class","prot_attr","sens_attr"
        ]
        
        # Also common demographic column names people use directly
        demographic_keywords = [
            "gender", "race", "ethnicity", "age_group",
            "religion", "nationality", "disability","sex","age","ethnic_group"
        ]
        
        found_explicit = False      # used recognized fairness terminology
        found_as_kwarg = False      # passed as sensitive_features= to a function
        found_demographic = False   # referenced demographic columns directly
        
        for node in ast.walk(self.tree):
            
            # Variable assignments: sensitive_features = df['gender']
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.lower() in sensitive_keywords:
                            found_explicit = True
            
            # Keyword arguments: fit(X, y, sensitive_features=A)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg and kw.arg.lower() in sensitive_keywords:
                        found_as_kwarg = True
            
            # Demographic column references: df['gender'], df["race"]
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                if node.slice.value.lower() in demographic_keywords:
                    found_demographic = True
        
        anchor = next(ast.walk(self.tree))
        weight = 15
        found_any = found_explicit or found_as_kwarg or found_demographic
        
        # Tiered scoring
        if found_explicit or found_as_kwarg:
            self.score += weight          # full credit — explicit fairness terminology
        elif found_demographic:
            self.score += weight * 0.5    # partial — demographic columns referenced but not labeled
        
        if found_any:
            details = []
            if found_explicit:
                details.append("sensitive features explicitly named")
            if found_as_kwarg:
                details.append("sensitive_features passed as argument")
            if found_demographic:
                details.append("demographic columns referenced")
            self.add_issue(anchor,
                f"FNA105: Sensitive feature identification detected ({'; '.join(details)}), "
                f"+{weight if (found_explicit or found_as_kwarg) else weight * 0.5}"
            )
        else:
            self.add_issue(anchor,
                "FNA105: No sensitive/protected features identified. "
                "Fairness analysis requires explicit identification of protected attributes "
                "(e.g., sensitive_features=df['gender'])."
            )
        
    def check_bias_mitigation_libraries(self):
        libs = ["aif360", "fairlearn", "equitas", "aequitas", "fairness_indicators","tensorflow_model_analysis"]
        
        # Step 1: track imports
        imported_libs = []
        fairness_imported_names = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".", 1)[0]
                    if root in libs and root not in imported_libs:
                        imported_libs.append(root)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module.split(".", 1)[0] if node.module else ""
                if mod in libs:
                    if mod not in imported_libs:
                        imported_libs.append(mod)
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        fairness_imported_names.append(local_name)
        
        # Step 2: check usage
        used_libs = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in fairness_imported_names:
                    for imp in ast.walk(self.tree):
                        if isinstance(imp, ast.ImportFrom):
                            mod = imp.module.split(".", 1)[0] if imp.module else ""
                            if mod in libs:
                                for alias in imp.names:
                                    local = alias.asname if alias.asname else alias.name
                                    if local == fn.id and mod not in used_libs:
                                        used_libs.append(mod)
                elif isinstance(fn, ast.Attribute):
                    root = fn
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in libs and root.id not in used_libs:
                        used_libs.append(root.id)
        
        anchor = next(ast.walk(self.tree))
        weight = 15
        deduction = 5
        
        if not imported_libs:
            self.add_issue(anchor,
                "FNA106: No bias mitigation library imported (e.g., aif360, fairlearn, aequitas)"
            )
            return
        
        imported_not_used = [l for l in imported_libs if l not in used_libs]
        
        if used_libs:
            # Library is used  award full credit, then deduct if there are ALSO unused imports
            self.score += weight
            if imported_not_used:
                self.score -= deduction
                self.add_issue(anchor,
                    f"FNA106: Bias mitigation used: {', '.join(used_libs)} (+{weight}), "
                    f"but also imported without use: {', '.join(imported_not_used)} (-{deduction})"
                )
            else:
                self.add_issue(anchor,
                    f"FNA106: Bias mitigation libraries imported and used: {', '.join(used_libs)}, +{weight}"
                )
        else:
            # Nothing used  no credit awarded, no deduction applied
            # (can't deduct from a check that gave 0 points — it would unfairly penalize the total)
            self.add_issue(anchor,
                f"FNA106: Bias mitigation libraries imported but never called: {', '.join(imported_not_used)}. "
                f"No credit awarded (would have been +{weight} if used)."
            )

    def check_fairness_metrics(self):
        # Known specific metric names 
        known_metrics = [
            "equalized_odds", "demographic_parity", "statistical_parity",
            "disparate_impact_ratio", "disparate_impact",
            "average_abs_odds_difference", "average_odds_difference",
            "consistency", "false_discovery_rate",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "error_rate_difference",
            "error_rate_ratio",
            "false_omission_rate_difference",
            "demographic_parity_difference",
            "demographic_parity_ratio",
            "true_positive_rate_difference",
            "false_positive_rate_difference",
            "selection_rate",
            "MetricFrame",                    # fairlearn metric container
            "ClassificationMetric",           # aif360 metric class
            "BinaryLabelDatasetMetric",       # aif360 metric class
            "SampleDistortionMetric",         # aif360 metric class
            "DatasetMetric",                  # aif360 base metric class
        ]
        
        # Submodules that contain fairness metrics specifically
        metric_submodules = [
            "aif360.metrics",
            "fairlearn.metrics",
            # aequitas
            "aequitas",              # top-level covers aequitas.group, aequitas.bias, aequitas.fairness
            "aequitas.group",
            "aequitas.bias",
            "aequitas.fairness",
            # fairness-indicators
            "fairness_indicators",
            "tensorflow_model_analysis",
        ]        
        # Broader fairness libs (for dotted-call detection)
        fairness_libs = ["aif360", "fairlearn", "equitas", "aequitas", "fairness_indicators","tensorflow_model_analysis"]
        
        fairness_keywords = [
            "parity", "disparity", "odds_difference", "odds_ratio",
            "demographic_parity", "equal_opportunity", "disparate_impact",
            "fairness_score", "fairness_metric", "bias_score"
        ]
        
        # Step 1: collect names imported from metric submodules specifically
        fairness_metric_names = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(sm) for sm in metric_submodules):
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        fairness_metric_names.append(local_name)
        
        found_specific = []
        found_lib_metric = False
        found_custom = False
        
        # Step 2: walk the tree looking for metric usage
        for node in ast.walk(self.tree):
            
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute)
                        else None)
                
                # Match against hardcoded known metric names
                if name and name in known_metrics and name not in found_specific:
                    found_specific.append(name)
                
                # Match against names imported from metric submodules
                if name and name in fairness_metric_names:
                    found_lib_metric = True
                    if name not in found_specific:
                        found_specific.append(name)
                
                # Detect dotted calls rooted in fairness libs
                # e.g. fairlearn.metrics.equalized_odds_difference(...)
                if isinstance(fn, ast.Attribute):
                    root = fn
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in fairness_libs:
                        found_lib_metric = True
            
            # Custom function definitions with fairness keywords
            elif isinstance(node, ast.FunctionDef):
                if any(kw in node.name.lower() for kw in fairness_keywords):
                    found_custom = True
            
            # Variables named with fairness keywords
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(kw in target.id.lower() for kw in fairness_keywords):
                            found_custom = True
        
        anchor = next(ast.walk(self.tree))
        weight = 10
        found_any = found_specific or found_lib_metric or found_custom
        
        # Tiered scoring
        if found_specific or found_lib_metric:
            self.score += weight
        elif found_custom:
            self.score += weight * 0.5
        
        if found_any:
            details = []
            if found_specific:
                details.append(f"known metrics: {', '.join(found_specific)}")
            if found_lib_metric:
                details.append("fairness library metric call detected")
            if found_custom:
                details.append("custom fairness metric implementation detected")
            self.add_issue(anchor,
                f"FNA107: Fairness metrics detected ({'; '.join(details)}), "
                f"+{weight if (found_specific or found_lib_metric) else weight * 0.5}"
            )
        else:
            self.add_issue(anchor,
                "FNA107: No fairness metrics found. Consider equalized_odds, "
                "demographic_parity, or metrics from fairlearn.metrics/aif360.metrics."
            )
         
    def check_model_training(self):
        # Known technique names (fallback)
        known_terms = [
            "adversarial", "reweighting", "DisparateImpactRemover",
            "AdversarialDebiasing", "ARTClassifier", "PrejudiceRemover",
            "EqOddsPostprocessing", "DeterministicReranking", "GerryFairClassifier",
            "ThresholdOptimizer", "ExponentiatedGradient",
            "MetaFairClassifier", "CalibratedEqOddsPostprocessing",
            "RejectOptionClassification", "LearnedFairRepresentations",
            "OptimizedPreprocessing"
        ]
        
        # Submodules that contain actual training/mitigation techniques
        training_submodules = [
            "aif360.algorithms",           # covers inprocessing, preprocessing, postprocessing
            "fairlearn.reductions",
            "fairlearn.postprocessing",
            "fairlearn.adversarial",
        ]
        
        # Broader fairness libs (for the .fit() caller check)
        fairness_libs = ["aif360", "fairlearn", "equitas", "aequitas", "fairness_indicators","tensorflow_model_analysis"]
        
        fairness_keywords = ["fair", "debias", "reweight", "mitigation", "equit"]
        
        # Step 1: collect names imported from TRAINING submodules specifically
        fairness_training_names = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(sm) for sm in training_submodules):
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        fairness_training_names.append(local_name)
        
        found_specific = []
        found_fairness_fit = False
        found_custom = False
        
        # Step 2: walk the tree looking for evidence of fairness-aware training
        for node in ast.walk(self.tree):
            
            # Detect calls to known/imported training techniques
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute)
                        else None)
                
                if name:
                    # Match against hardcoded known technique names
                    for t in known_terms:
                        if t in name and t not in found_specific:
                            found_specific.append(t)
                    
                    # Match against names imported from training submodules
                    if name in fairness_training_names and name not in found_specific:
                        found_specific.append(name)
                
                # Detect .fit() called on a fairness training object
                if isinstance(fn, ast.Attribute) and fn.attr == "fit":
                    caller = fn.value
                    if isinstance(caller, ast.Name):
                        # Walk back to find the assignment of this variable
                        for assign_node in ast.walk(self.tree):
                            if (isinstance(assign_node, ast.Assign)
                                    and any(isinstance(t, ast.Name) and t.id == caller.id
                                            for t in assign_node.targets)
                                    and isinstance(assign_node.value, ast.Call)):
                                call_fn = assign_node.value.func
                                call_name = (call_fn.id if isinstance(call_fn, ast.Name)
                                             else call_fn.attr if isinstance(call_fn, ast.Attribute)
                                             else None)
                                
                                # Caller was instantiated from a training-submodule import
                                if call_name and call_name in fairness_training_names:
                                    found_fairness_fit = True
                                
                                # Or instantiated via dotted call on a training submodule
                                # e.g. aif360.algorithms.inprocessing.AdversarialDebiasing(...)
                                if isinstance(call_fn, ast.Attribute):
                                    root = call_fn
                                    while isinstance(root, ast.Attribute):
                                        root = root.value
                                    if isinstance(root, ast.Name) and root.id in fairness_libs:
                                        found_fairness_fit = True
            
            # Detect custom function definitions with fairness keywords
            elif isinstance(node, ast.FunctionDef):
                if any(kw in node.name.lower() for kw in fairness_keywords):
                    found_custom = True
        
        anchor = next(ast.walk(self.tree))
        weight = 10
        found_any = found_specific or found_fairness_fit or found_custom
        
        # Tiered scoring
        if found_specific or found_fairness_fit:
            self.score += weight
        elif found_custom:
            self.score += weight * 0.5
        
        if found_any:
            details = []
            if found_specific:
                details.append(f"known techniques: {', '.join(found_specific)}")
            if found_fairness_fit:
                details.append("fairness training object trained with .fit()")
            if found_custom:
                details.append("custom fairness-aware implementation detected")
            self.add_issue(anchor,
                f"FNA108: Fairness-aware training detected ({'; '.join(details)}), "
                f"+{weight if (found_specific or found_fairness_fit) else weight * 0.5}"
            )
        else:
            self.add_issue(anchor,
                "FNA108: No fairness-aware training detected. Consider using techniques like "
                "reweighting, adversarial debiasing, or fairness-constrained optimization "
                "(from aif360.algorithms or fairlearn.reductions)."
            )

    def check_evaluation(self):
        known_eval_funcs = [
            "audit_bias", "classification_report", "confusion_matrix",
            "roc_auc_score", "f1_score", "evaluate", "model_report",
            # aequitas
            "Audit", "Auditor",
            # fairness-indicators evaluation
            "FairnessIndicators", "run_model_analysis",
        ]
        # Submodules that specifically indicate evaluation/auditing work
        eval_submodules = [
            "aequitas.audit",
            "tensorflow_model_analysis",
        ]
        eval_keywords = ["eval", "audit", "report", "assess", "review", "benchmark"]
    
        # Step 1: collect names imported from evaluation submodules
        eval_imported_names = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(sm) for sm in eval_submodules):
                    for alias in node.names:
                        local_name = alias.asname if alias.asname else alias.name
                        eval_imported_names.append(local_name)
    
        found_specific = []
        found_lib_eval = False
        found_custom = False
    
        # Step 2: walk the tree for evaluation evidence
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute)
                        else None)
    
                # Match against hardcoded known evaluation functions
                if name and name in known_eval_funcs and name not in found_specific:
                    found_specific.append(name)
    
                # Match against names imported from evaluation submodules
                if name and name in eval_imported_names:
                    found_lib_eval = True
                    if name not in found_specific:
                        found_specific.append(name)
    
                # Detect dotted calls rooted in evaluation submodule namespaces
                # e.g. aequitas.audit.Audit(...) or tensorflow_model_analysis.run_model_analysis(...)
                if isinstance(fn, ast.Attribute):
                    root = fn
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id in ("aequitas", "tensorflow_model_analysis", "tfma"):
                        found_lib_eval = True
    
            elif isinstance(node, ast.FunctionDef):
                if any(kw in node.name.lower() for kw in eval_keywords):
                    found_custom = True
    
        anchor = next(ast.walk(self.tree))
        weight = 10
        found_any = found_specific or found_lib_eval or found_custom
    
        # Tiered scoring — matches the pattern in metrics/training
        if found_specific or found_lib_eval:
            self.score += weight
        elif found_custom:
            self.score += weight * 0.5
    
        if found_any:
            details = []
            if found_specific:
                details.append(f"found: {', '.join(found_specific)}")
            if found_lib_eval:
                details.append("evaluation library call detected")
            if found_custom:
                details.append("custom evaluation function detected")
            self.add_issue(anchor,
                f"FNA109: Evaluation detected ({'; '.join(details)}), "
                f"+{weight if (found_specific or found_lib_eval) else weight * 0.5}"
            )
        else:
            self.add_issue(anchor,
                "FNA109: No fairness evaluation found (e.g., audit_bias, "
                "classification_report, aequitas.Audit, or a custom evaluation function)"
            )