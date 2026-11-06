"""Article-bounded causal windows; every within-article target scored once."""
def articles_from_rows(rows):
    articles=[];current=[];start=0
    for index,text in enumerate(rows):
        if not isinstance(text,str):raise ValueError('Expected text rows')
        if not text.strip():continue
        parts=text.strip().split()
        heading=(len(parts)>2 and parts[0]==parts[-1]=='=' and parts[1]!='=' and parts[-2]!='=')
        if heading and current:
            articles.append({'source_row_start':start,'text':'\n'.join(current)})
            current=[]
        if not current:start=index
        current.append(text)
    if current:articles.append({'source_row_start':start,'text':'\n'.join(current)})
    return articles


def causal_windows(token_count,max_context=1024,stride=512):
    if type(token_count) is not int or token_count<0 or not 1<=stride<max_context:
        raise ValueError('Valid token count and 1<=stride<context required')
    result=[]
    for target_start in range(1,token_count,stride):
        stop=min(target_start+stride,token_count)
        begin=max(0,stop-max_context)
        assert begin<target_start<stop and stop-begin<=max_context
        result.append({'context_start':begin,'context_stop':stop,
                       'target_start':target_start,'target_stop':stop})
    return result
