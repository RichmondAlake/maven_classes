"""Preserve explanations when variant adapters are inserted before existing cells."""
import ast
import nbformat

EXPLANATIONS = {
    'update_profile': 'Accept extracted preferences and a source turn ID. Validate typed fields and source ownership, merge current facts with the version read from Oracle, then publish the framework entity profile. Source transcripts remain available after corrections.',
    'extract_entities': 'Accept the user text and call the raw Anthropic client to extract only explicitly stated durable preferences. Dates and supplier claims do not become personal preferences. The decision edition can gate this call with the preceding entity detector.',
    'workflow_memory': 'Read the most recent real execution steps for this owner and thread. record_step stores the selected action, successful or failed outcome and the usage belonging to that action decision. The loop reads these records again before its next call.',
    'compact_thread': 'Accept the number of newest active turns to keep and an optional quality gate. Summarize actual older source turns, check quality before marking them compacted, preserve the originals and publish a discoverable archive pointer.',
}


def ensure_explanations(cells):
    result=[]
    for cell in cells:
        if cell.cell_type == 'code' and result and result[-1].cell_type == 'code':
            methods=[node for node in ast.parse(cell.source).body if isinstance(node,ast.FunctionDef)]
            name=methods[0].name if methods else 'Inspect the next result'
            text=EXPLANATIONS.get(name)
            if text is None:
                raise ValueError(f'Write an educational explanation for {name}.')
            parameters=', '.join(arg.arg for arg in methods[0].args.args)
            result.append(nbformat.v4.new_markdown_cell(f'### {name}\n\n{text}\n\n**Arguments:** {parameters}.'))
        result.append(cell)
    return result
