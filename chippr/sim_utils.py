def ingest(in_info):
    """
    Function reading in parameter file to define functions necessary for generation of posterior probability distributions

    Parameters
    ----------
    in_info: string or dict
        string containing path to plaintext input file or dict containing likelihood input parameters

    Returns
    -------
    in_dict: dict
        dict containing keys and values necessary for posterior probability distributions
    """
    if type(in_info) == str:
        with open(loc) as infile:
            lines = (line.split(None) for line in infile)
            in_dict = {defn[0]:defn[1:] for defn in lines}
    else:
        in_dict = in_info
    return in_dict

def lf_params(in_info):
    """
    Function reading in parameter file to define likelihood function

    Parameters
    ----------
    in_info: string or dict
        string containing path to plaintext input file or dict containing likelihood input parameters

    Returns
    -------
    out_dict: dict
        dict containing keys and values necessary for likelihood function definition
    """
    in_dict = ingest(in_info)

    out_dict = {}
    if 'sigma' in in_dict:
        out_dict['sigma'] = in_dict['sigma']
    else:
        out_dict['sigma'] = 0.1

    return out_dict

def int_pr_params(in_info):
    """
    Function reading in parameter file to define interim prior function

    Parameters
    ----------
    in_info: string or dict
        string containing path to plaintext input file or dict containing interim prior input parameters

    Returns
    -------
    out_dict: dict
        dict containing keys and values necessary for interim prior function definition
    """
    in_dict = ingest(in_info)

    out_dict = {}
    if 'intp' in in_dict:
        out_dict['int_pr'] = in_dict['int_pr']
    else:
        out_dict['int_pr'] = 'flat'

    return out_dict