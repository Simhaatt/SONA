# Test-time adaptation: frozen vs online (VoiceBank vs MS-SNSD)

| label | method | pesq | si_sdr_i | stoi | estoi | lsd |
| --- | --- | --- | --- | --- | --- | --- |
| VoiceBank frozen | neural_attention | 2.5012 | 9.0817 | 0.9322 | 0.8344 | 5.9701 |
| VoiceBank strict-guard online | attention_online_lowrank_s1_a2_d0p01_go0p0001 | 2.5034 | 9.0790 | 0.9322 | 0.8344 | 5.9630 |
| MS-SNSD frozen | neural_attention | 1.3232 | 2.5861 | 0.8035 | 0.6643 | 11.9075 |
| MS-SNSD unguarded online d0.01 | attention_online_lowrank_s1_a2_d0p01 | 1.3138 | 2.6003 | 0.8032 | 0.6635 | 11.9681 |
| MS-SNSD unguarded online d0.02 | attention_online_lowrank_s1_a2_d0p02 | 1.3074 | 2.6092 | 0.8028 | 0.6628 | 12.0173 |
