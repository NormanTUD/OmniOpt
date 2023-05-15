use strict;
use warnings;
use autodie;
use Data::Dumper;

my %options = (
        debug => 0,
        config_path => undef,
        delete_dimension => []
);


sub error (@) {
        for (@_) {
                warn "$_\n";
        }
        exit(1);
}

sub debug (@) {
        return unless $options{debug};
        for (@_) {
                warn "$_\n";
        }
}

sub _help {
        my $exit_code = shift;

        print <<EOF;
--help                          This help
--debug                         Enables debug options
--config_path=PATH              Path to a config file
--delete_dimension=NAME         Deletes the dimension with the name NAME and reduce the number of dimensions and the id of following dimensions,
                                can be used multiple times to delete multiple dimensions
EOF
        exit($exit_code);
}

sub analyze_args {
        for(@_) {
                if(/^--debug$/) {
                        $options{debug} = 1;
                } elsif (/^--config_path=(.*)$/) {
                        $options{config_path} = $1;
                } elsif (/^--delete_dimension=(.*)$/) {
                        push @{$options{delete_dimension}}, $1;
                } elsif (/^--help$/) {
                        _help(0);
                } else {
                        warn "Unknown parameter $_";
                        _help(1);
                }
        }

        die "No (valid) --config_path given" if(!$options{config_path} || !-e $options{config_path});
}

analyze_args(@ARGV);

sub read_ini_file {
        my $file = shift;
        debug("read_ini_file($file)");
        debug "read_ini_file($file)";
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
        return $conf;
}

sub get_dim_by_name {
        my $name = shift;

        $name =~ s#\s*=.*##;
        $name =~ s#(?:.*_)?(\d+)(?:_.*)?#$1#g;

        return $name;
}

sub replace_dim {
        my $str = shift;
        my $old_dim = shift;
        my $new_dim = shift;

        $str =~ s#$old_dim#$new_dim#g;

        return $str;
}

sub hash_to_ini {
        my $hashref = shift;
        my %hash = %{$hashref};

        my $file = "";

        my $i = 0;
        foreach my $k (keys(%hash)) {
                if($i != 0) {
                    $file .= "\n";
                }
                $file .= "[$k]\n";
                my @lines = ();
                foreach my $value_key (keys(%{$hash{$k}})) {
                        my $value_value = $hash{$k}{$value_key};
                        if($value_key =~ /dimensions/) {
                                $file .= "$value_key = $value_value\n";
                        } else {
                                push @lines, "$value_key = $value_value\n";
                        }
                }

                if($k eq "DIMENSIONS") {
                        @lines = sort { 
                                return get_dim_by_name($a) <=> get_dim_by_name($b)
                        } @lines;
                }

                $file .= join("", @lines);
                $i++;
        }
        return $file;
}

sub main {
        debug "main";
        my $file = read_ini_file($options{config_path});

        foreach my $deldim (@{$options{delete_dimension}}) {
                my $dim_nr = undef;
                foreach my $dim (keys(%{$file->{DIMENSIONS}})) {
                        my $val = $file->{DIMENSIONS}->{$dim};
                        if($dim =~ m#dim_(\d+)_name# && $val eq $deldim) {
                                $dim_nr = get_dim_by_name($dim);
                        }
                }

                if(defined $dim_nr) {
                        my %tmp;
                        foreach my $dim (keys(%{$file->{DIMENSIONS}})) {
                                if($dim !~ /^(.*_)?${dim_nr}(?:_.*)?$/) {
                                        my $val = $file->{DIMENSIONS}->{$dim};
                                        $tmp{$dim} = $val;
                                }
                        }

                        foreach my $dim (sort { $a cmp $b } keys(%{$file->{DIMENSIONS}})) {
                                if($dim !~ "dimensions") {
                                        my $tdim_nr = get_dim_by_name($dim);
                                        my $original_tdim_nr = $tdim_nr;
                                        if($tdim_nr > $dim_nr) {
                                                my $val = $tmp{$dim};
                                                delete $tmp{$dim};
                                                $tdim_nr--;
                                                $tmp{replace_dim($dim, $original_tdim_nr, $tdim_nr)} = $val;
                                        }
                                }
                        }

                        $tmp{dimensions}--;
                        $file->{"DIMENSIONS"} = \%tmp;
                }
        }

        print hash_to_ini($file);
        #die Dumper $file;
}

main();
