import os
import re

REPORTS_DIR = '/opt/data/hermes_workspace/bnpl_ml_project/results/model_comparison'
SUMMARY_PATH = os.path.join(REPORTS_DIR, 'SUMMARY.md')

# Pattern to extract metrics from a report file
# Example lines:
# - AUC-PR: 0.4523
# - Precision: 0.7704
# - Recall: 0.7891
# - F1-Score: 0.7354
pattern = re.compile(r'-\s*(AUC-PR|Precision|Recall|F1-Score):\s*([\d.]+)')

results = []

for filename in os.listdir(REPORTS_DIR):
    if filename.startswith('report_') and filename.endswith('.md') and filename != 'SUMMARY.md':
        filepath = os.path.join(REPORTS_DIR, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract model and config from filename
        # filename format: report_{config}_{model}.md
        # Remove 'report_' and '.md'
        middle = filename[7:-3]
        # Split by last underscore? Actually, we know the format is config_model
        # But config and model might have underscores. We'll split by the last underscore.
        # However, in our naming, we used underscores in config and model, but we can split by the first underscore after 'report_'?
        # Let's do a simple split: we know the pattern is report_CONFIG_MODEL.md
        # We'll split the middle part by the last underscore.
        if '_' in middle:
            # Find the last underscore
            last_underscore = middle.rfind('_')
            config = middle[:last_underscore]
            model = middle[last_underscore+1:]
        else:
            config = middle
            model = ''
        
        # Extract metrics
        metrics = {}
        for match in pattern.finditer(content):
            key = match.group(1)
            value = float(match.group(2))
            metrics[key] = value
        
        # We expect all four metrics
        if len(metrics) == 4:
            results.append({
                'config': config,
                'model': model,
                'auc_pr': metrics['AUC-PR'],
                'precision': metrics['Precision'],
                'recall': metrics['Recall'],
                'f1': metrics['F1-Score'],
                'filename': filename
            })

# Sort by AUC-PR descending
results.sort(key=lambda x: x['auc_pr'], reverse=True)

# Write summary
with open(SUMMARY_PATH, 'w') as f:
    f.write('# BNPL V2 Model Experiment Summary (Collected from Existing Reports)\n\n')
    f.write(f'**Total Reports Processed**: {len(results)}\n\n')
    f.write('## Top Models by AUC-PR\n\n')
    f.write('| Rank | Configuration | Model | AUC-PR | Precision | Recall | F1 |\n')
    f.write('|------|---------------|-------|--------|-----------|--------|----|\n')
    for i, res in enumerate(results):
        f.write(f"| {i+1} | {res['config']} | {res['model']} | {res['auc_pr']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1']:.4f} |\n")
    
    f.write('\n## Best Overall Model\n\n')
    if results:
        best = results[0]
        f.write(f'- **Configuration**: {best["config"]}\n')
        f.write(f'- **Model**: {best["model"]}\n')
        f.write(f'- **AUC-PR**: {best["auc_pr"]:.4f}\n')
        f.write(f'- **Precision**: {best["precision"]:.4f}\n')
        f.write(f'- **Recall**: {best["recall"]:.4f}\n')
        f.write(f'- **F1-Score**: {best["f1"]:.4f}\n\n')
    else:
        f.write('No valid reports found.\n')

    f.write('\n## All Reports\n')
    f.write(f'Individual reports are stored in the `{REPORTS_DIR}` directory.\n')
    f.write(f'Total reports processed: {len(results)}\n')

print(f'Summary written to {SUMMARY_PATH}')
print(f'Processed {len(results)} reports.')
if results:
    print(f'Best AUC-PR: {results[0]["auc_pr"]:.4f} ({results[0]["config"]} + {results[0]["model"]})')