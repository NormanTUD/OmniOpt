use strict;
use warnings;
use Term::ANSIColor;
use Data::Dumper;

sub error (@) {
        foreach (@_) {
                warn color('red')."ERROR:\t\t$_.".color('reset')."\n";
        }
        exit 1;
}



sub read_ini_file {
        my $file = shift;
        my $conf;
        open (my $INI, $file) || error "Can't open ini-file `$file`: $!\n";
        my $section = '';
        while (<$INI>) {
                chomp;
                if (/^\s*\[\s*(.+?)\s*\]\s*$/) {
                        $section = $1;
                }

                if ( /^\s*([^=]+?)\s*=\s*(.*?)\s*$/ ) {
                        $conf->{$section}->{$1} = $2;
                }
        }
        close ($INI);
        return %{$conf};
}

sub pad_space_right {
        my $str = shift // "";
        my $length = shift;
        my $char = shift // ' ';

        my $missing = $length - length(remove_color_codes($str));

        return $str.($char x $missing);
}

sub sum {
        my $sum = 0;
        foreach (@_) {
                $sum += $_;
        }
        return $sum;
}

sub remove_color_codes {
        my $str = shift;
        return $str unless $str;
        $str =~ s/\e\[[0-9;]*m(?:\e\[K)?//g;
        return $str;
}

sub get_longest_line_length {
        my $len = 0;
        foreach (@_) {
                if(length(remove_color_codes($_)) > $len) {
                        $len = length(remove_color_codes($_));
                }
        }
        return $len;
}

sub array_to_table {
        my @array = @_;
        my $str = '';

        my @column_widths = ();

        my $number_of_cols = 0;

        foreach my $line (@array) {
                if(ref $line) {
                        if(@$line > 1) {
                                foreach my $item_id (0 .. scalar @{$line} - 1) {
                                        my $str = remove_color_codes($line->[$item_id]);
                                        if(!exists $column_widths[$item_id] || ($str && length($str) > $column_widths[$item_id])) {
                                                $column_widths[$item_id] = length($str);
                                                $number_of_cols = $item_id + 1 if $item_id + 1 > $number_of_cols;
                                        }
                                }
                        }
                }
        }

        my $max_line_length = sum(@column_widths);

        my @table_lines = ();
        if (@array) {
                foreach my $line (@array) {
                        my $this_str = '';
                        if(ref $line) {
                                if (scalar @{$line} == 1) {
                                        $this_str .= "| ".pad_space_right($line->[0], $max_line_length + ($number_of_cols * 3) - 1);
                                } else {
                                        foreach my $item_id (0 .. scalar @{$line} - 1) {
                                                $this_str .= (($item_id == 0 ? '' : ' ')."| ".pad_space_right($line->[$item_id], $column_widths[$item_id] + 1));
                                        }
                                }
                                push @table_lines, $this_str." |";
                        } else {
                                push @table_lines, $line;
                        }
                }
        } else {
                print "Empty array";
        }

        my $longest_line = get_longest_line_length(@table_lines);
        my ($startsign, $endsign) = ('', '');
        foreach my $i (0 .. $#table_lines) {
                if($table_lines[$i] eq '-') {
                        if($i == $#table_lines) {
                                ($startsign, $endsign) = ('\\', '/');
                        } elsif($i == 0) {
                                ($startsign, $endsign) = ('/', '\\');
                        } else {
                                ($startsign, $endsign) = ('|', '|');
                        }
                        $table_lines[$i] = $startsign.('-' x ($longest_line - 2)).$endsign;

                }
        }

        $str = join("\n", @table_lines)."\n";

        return $str;
}

sub create_str_from_ini {
        my $project_name = shift;
        my %config = @_;

        my @lines = ();

        print color("red")."Press 'q' to exit and return to run-evaluate.sh".color("reset")."\n";

        my ($yellow, $reset) = (color("yellow"), color("reset"));

        my $objective_program = $config{DATA}{objective_program};
        my $num_of_params = $config{DIMENSIONS}{dimensions};
        my $max_evals = $config{DATA}{max_evals};

        push @lines, "-";
        push @lines, ["Project name", "$yellow$project_name$reset"];
        push @lines, "-";
        push @lines, ["Objective program", $objective_program];
        push @lines, ["Number of hyperparameters", $num_of_params];
        push @lines, ["Max. evals", $max_evals];
        foreach my $this_param_nr (0 .. $num_of_params - 1) {
                push @lines, "-";
                my $name = $config{DIMENSIONS}{"dim_${this_param_nr}_name"};
                my $range_generator = $config{DIMENSIONS}{"range_generator_$this_param_nr"};
                push @lines, ["$yellow$name$reset (\$x_$this_param_nr), range_generator: $yellow$range_generator$reset"];
                if($range_generator eq "hp.randint") {
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, ["Maximum number", $max];
                } elsif($range_generator eq "hp.choice") {
                        my $choice = $config{DIMENSIONS}{"options_${this_param_nr}"};
                        push @lines, ["Items", $choice];
                } elsif($range_generator eq "hp.uniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, ["Minimum number", $min];
                        push @lines, ["Maximum number", $max];
                } elsif($range_generator eq "hp.quniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, ["Minimum number", $min];
                        push @lines, ["Maximum number", $max];
                        push @lines, ["q", $q];
                } elsif($range_generator eq "hp.loguniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, ["Minimum number", $min];
                        push @lines, ["Maximum number", $max];
                } elsif($range_generator eq "hp.qloguniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, ["Minimum number", $min];
                        push @lines, ["Maximum number", $max];
                        push @lines, ["q", $q];
                } elsif($range_generator eq "hp.normal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        push @lines, ["Mu", $mu];
                        push @lines, ["Sigma", $sigma];
                } elsif($range_generator eq "hp.qnormal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, ["Mu", $mu];
                        push @lines, ["Sigma", $sigma];
                        push @lines, ["q", $q];
                } elsif($range_generator eq "hp.lognormal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        push @lines, ["Mu", $mu];
                        push @lines, ["Sigma", $sigma];
                } elsif($range_generator eq "hp.qlognormal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, ["Mu", $mu];
                        push @lines, ["Sigma", $sigma];
                        push @lines, ["q", $q];
                } else {
                        die $range_generator;
                }
        }
        push @lines, "-";

        #die Dumper %config;

        return array_to_table(@lines);
}

my $project_name = shift @ARGV;
my $ini_file = shift @ARGV;
print create_str_from_ini($project_name, read_ini_file($ini_file));
