use strict;
use warnings;
use feature 'say';
use Data::Dumper;
use Term::ANSIColor;
use lib '.';

use Getopt::Long;
my %options = (
        filename => undef,
        debug => 0
);
GetOptions (
    "filename=s"        => \$options{filename},
    "debug"             => \$options{debug}
) or die("Error in command line arguments\n");

sub debug (@) {
        return if !$options{debug};
        foreach (@_) {
            my $str = $_;
            if(ref $_) {
                $str = Dumper $_;
            } else {
                $str = "$_\n";
            }

            my $green = color("green underline");
            my $reset = color("reset");
            my $DEBUG = "DEBUG";
            $str =~ s#^|\n#\n$green$DEBUG$reset: #gis;
            warn $str;
        }
}

debug \%options;

main(%options);

sub main {
        my %par = (
            filename => undef,
            debug => 0,
            @_
        );

        debug "Inside main";

        if($par{filename}) {
                debug "Filename given: $par{filename}";
                my $i = 0;
                my $max = 30;

                while ($i != $max) {
                    if(!-e $par{filename}) {
                        warn "$par{filename} not found, sleeping for a second before trying again ($i out of $max)";
                        sleep 1;
                    }
                    $i++;
                }

                if(-e $par{filename}) {
                    debug "File found: $par{filename}";
                    my $text = shift;
                    open my $fh, '<', $par{filename} or die $!;
                    while (<$fh>) {
                        $text .= $_;
                    }
                    close $fh;
                    foreach my $this_text (split(/^=======================$/, $text)) {
                        print analyze_string($this_text);
                    }
                } else {
                    warn "File NOT found: $par{filename}";
                    exit(1)
                }
        } else {
                warn "Filename not given";
        }
}

sub analyze_string {
    debug "analyze_string";
    my $text = shift;
    my %epochendaten = ();

    while ($text =~ m#Epoch (\d+)/(\d+).*?(\d+)us/step - loss: (\d+\.\d+) - acc: (\d+\.\d+)#gis) {
        my $epoche = $1;
        my $von_epochen = $2;
        my $usprostep = $3;
        my $loss = $4;
        my $acc = $5;
        $epochendaten{gesamtanzahlepochen} = $von_epochen;
        $epochendaten{$epoche} = {
            usprostep => $usprostep,
            loss => $loss,
            acc => $acc
        };
        debug Dumper $epochendaten{$epoche};
        #print ":::EPOCHE $1 von $2 ($usprostep us pro schritt), loss = $loss, acc = $acc\n";
    }

    my $json = to_json(\%epochendaten);

    return $json;
}


sub to_json {
    my $val = shift;
    if (not defined $val) {
        return "null";
    } elsif (not ref $val) {
        $val =~ s/([\0-\x1f\"\\])/sprintf "\\u%04x", ord $1/eg;
        return '"' . $val . '"';
    } elsif (ref $val eq 'ARRAY') {
        return '[' . join(',', map to_json($_), @$val) . ']';
    } elsif (ref $val eq 'HASH') {
        return '{' . join(',', map to_json($_) . ":" . to_json($val->{$_}), sort keys %$val) . '}';
    } else {
        die "Cannot encode $val as JSON!\n";
    }
}
