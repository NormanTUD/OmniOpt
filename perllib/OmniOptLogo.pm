use strict;
use warnings;
use Term::ANSIColor;

sub show_logo {
	print color("bold underline on_green").'OmniOpt'.color("reset")."\n";
	return;
}

1;
