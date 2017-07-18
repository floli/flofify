import errno, os

def norm_path(*parts):
    """ Returns the normalized, absolute, expanded and joint path, assembled of all parts. """
    parts = [ str(p) for p in parts ] 
    return os.path.abspath(os.path.expanduser(os.path.join(*parts)))


def XDG_DATA_HOME(*append):
    """ Returns data home (usually ~/.local/share) and joins/appends arguments to it. """
    try:
        data = os.environ["XDG_DATA_HOME"]
        if data == "":
            raise KeyError
    except KeyError:
        data = norm_path("~/.local/share")
    return norm_path(data, *append)

def XDG_CONFIG_HOME(*append):
    """ Returns config home (usually ~/.config/share) and joins/appends arguments to it. """
    try:
        config = os.environ["XDG_CONFIG_HOME"]
        if config == "":
            raise KeyError
    except KeyError:
        config = norm_path("~/.config")
    return norm_path(config, *append)



def mkpath(path):
    """ Creates the entire path.
    See http://bugs.python.org/issue13498 why os.makedirs is not suitable. """
    try:
        os.makedirs(path, exist_ok = True)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass # Exception is raised due to mode difference
        else:
            raise # Should not happen
