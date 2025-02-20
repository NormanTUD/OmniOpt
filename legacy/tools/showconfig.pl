use strict;
use warnings;
use Term::ANSIColor;
use Data::Dumper;

my $project_name = shift @ARGV;
my $ini_file = shift @ARGV;
my $die_on_undef = shift @ARGV // 0;

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

sub die_on_undef ($) {
        my $arrayref = shift;
        if(ref $arrayref eq "ARRAY") {
                if($die_on_undef) {
                        foreach my $item (@{$arrayref}) {
                                if(!defined $item || $item eq "") {
                                        die "Item undefined/empty. Description: ".(join(", ", grep { length($_) } @$arrayref));
                                }
                        }
                }
        } else {
                die "A";
        }
        return $arrayref;
}

sub get_number_from_x_string {
        my $x_string = shift;

        my $nr = undef;
        if($x_string =~ m#\(\$x_(\d+)\)#) {
                $nr = $1;
        }

        return $nr;
}

sub describe_x {
        my $str = shift;
        my $arrow = shift;
        my %config = @_;

        my $desc = "";

        my @splitted = split(//, $str);

        my %x_pos = ();

        while ($str =~ m#\s*?((?:--?[a-z0-9_]*?=|\s*?)?(?:int)?\(\$x_\d*\))#gism) {
                my $match = $1;
                $match =~ s#^\s##g;
                my $pos = pos($str);
                my $start = $pos - length($match);
                my $end = $pos + length($match);

                my $nr = get_number_from_x_string($match);
                my $name = $config{DIMENSIONS}{"dim_${nr}_name"};

                if(length($name) > length($match)) {
                        $name = "…".substr($name, length($name) - length($match) + 3, length($match));
                } elsif (length($name) < length($match)) {
                        my $j = 0;
                        while (length($name) < length($match)) {
                                if($j % 2 == 1) {
                                        $name .= " ";
                                } else {
                                        $name = " $name";
                                }
                                $j++;
                        }
                }

                $x_pos{$start} = { pos => $pos, match => $match, end => $end, nr => $nr, name => $name };
        }

        my $i = 0;
        while ($i <= length($str) - 1) {
                if(exists $x_pos{$i}) {
                        if($arrow) {
                                $desc .= "^" x length($x_pos{$i}{name});
                        } else {
                                $desc .= $x_pos{$i}{name};
                        }
                        $i += length($x_pos{$i}{name});
                } else {
                        $desc .= " ";
                        $i++;
                }
        }

        return $desc;
}

sub create_str_from_ini {
        my $project_name = shift;
        my $config_file_path = shift;
        my %config = read_ini_file($config_file_path);

        my @lines = ();

        #print color("red")."Press 'q' to exit and return to run-evaluate.sh".color("reset")."\n";

        my ($yellow, $reset) = (color("yellow"), color("reset"));

        my $objective_program = $config{DATA}{objective_program};
        my $num_of_params = $config{DIMENSIONS}{dimensions};
        my $max_evals = $config{DATA}{max_evals};

        push @lines, "-";
        push @lines, die_on_undef ["Project name", "$yellow$project_name$reset"];
        push @lines, "-";
        push @lines, die_on_undef ["Path of config.ini", $config_file_path];
        push @lines, die_on_undef ["Max. evals", $max_evals];
        push @lines, die_on_undef ["Objective program", $objective_program];
        push @lines, die_on_undef [" ", describe_x($objective_program, 1, %config)];
        push @lines, die_on_undef [" ", describe_x($objective_program, 0, %config)];
        push @lines, die_on_undef ["Number of hyperparameters", $num_of_params];
        foreach my $this_param_nr (0 .. $num_of_params - 1) {
print "$this_param_nr\n";
                push @lines, "-";
                my $name = $config{DIMENSIONS}{"dim_${this_param_nr}_name"};

                my $range_generator = $config{DATA}{range_generator_name};
                my $this_range_generator_name = "range_generator_$this_param_nr";
                if(exists $config{DIMENSIONS}{$this_range_generator_name}) {
                    $range_generator = $config{DIMENSIONS}{$this_range_generator_name};
                }

                push @lines, die_on_undef ["$yellow$name$reset (\$x_$this_param_nr), range_generator: $yellow$range_generator$reset"];
                if($range_generator eq "hp.randint") {
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, die_on_undef ["Maximum number", $max];
                } elsif($range_generator eq "hp.choice") {
                        my $choice = $config{DIMENSIONS}{"options_${this_param_nr}"};
                        $choice =~ s#\s*,\s*#,#g;
                        my @s = split /\s*,\s*/, $choice;
                        my $min = $s[0];
                        my $max = $s[$#s];

                        if($min > $max) {
                                ($max, $min) = ($min, $max);
                        }

                        if($choice eq join(",", $min .. $max)) {
                            my $new_choice = "$min, ..., $max (step-size: 1)";
                            if(length($new_choice) < length($choice)) {
                                $choice = $new_choice;
                            }
                        } else {
                        }


                        push @lines, die_on_undef ["Items", $choice];
                } elsif($range_generator eq "hp.uniformint") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, die_on_undef ["Minimum number", $min];
                        push @lines, die_on_undef ["Maximum number", $max];
                } elsif($range_generator eq "hp.pchoice") {
                        my $choice = $config{DIMENSIONS}{"options_${this_param_nr}"};
                        push @lines, die_on_undef ["Items", $choice];
                } elsif($range_generator eq "hp.choiceint") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, die_on_undef ["Minimum number", $min];
                        push @lines, die_on_undef ["Maximum number", $max];
                } elsif($range_generator eq "hp.uniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, die_on_undef ["Minimum number", $min];
                        push @lines, die_on_undef ["Maximum number", $max];
                } elsif($range_generator eq "hp.quniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, die_on_undef ["Minimum number", $min];
                        push @lines, die_on_undef ["Maximum number", $max];
                        push @lines, die_on_undef ["q", $q];
                } elsif($range_generator eq "hp.loguniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        push @lines, die_on_undef ["Minimum number", $min];
                        push @lines, die_on_undef ["Maximum number", $max];
                } elsif($range_generator eq "hp.qloguniform") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, die_on_undef ["Minimum number", $min];
                        push @lines, die_on_undef ["Maximum number", $max];
                        push @lines, die_on_undef ["q", $q];
                } elsif($range_generator eq "hp.normal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        push @lines, die_on_undef ["Mu", $mu];
                        push @lines, die_on_undef ["Sigma", $sigma];
                } elsif($range_generator eq "hp.qnormal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, die_on_undef ["Mu", $mu];
                        push @lines, die_on_undef ["Sigma", $sigma];
                        push @lines, die_on_undef ["q", $q];
                } elsif($range_generator eq "hp.lognormal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        push @lines, die_on_undef ["Mu", $mu];
                        push @lines, die_on_undef ["Sigma", $sigma];
                } elsif($range_generator eq "hp.choicestep") {
                        my $min = $config{DIMENSIONS}{"min_dim_${this_param_nr}"};
                        my $max = $config{DIMENSIONS}{"max_dim_${this_param_nr}"};
                        my $step = $config{DIMENSIONS}{"step_dim_${this_param_nr}"};
                        push @lines, die_on_undef ["Min", $min];
                        push @lines, die_on_undef ["Max", $max];
                        push @lines, die_on_undef ["Step", $step];
                } elsif($range_generator eq "hp.qlognormal") {
                        my $mu = $config{DIMENSIONS}{"mu_${this_param_nr}"};
                        my $sigma = $config{DIMENSIONS}{"sigma_${this_param_nr}"};
                        my $q = $config{DIMENSIONS}{"q_${this_param_nr}"};
                        push @lines, die_on_undef ["Mu", $mu];
                        push @lines, die_on_undef ["Sigma", $sigma];
                        push @lines, die_on_undef ["q", $q];
                } else {
                        die $range_generator;
                }
        }
        push @lines, "-";

        return array_to_table(@lines);
}

print create_str_from_ini($project_name, $ini_file);
