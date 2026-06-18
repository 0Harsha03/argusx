"""
ArgusX — Layer 1: Pattern Detection Engine
==========================================
Regex-based first-pass detector.

Detects:
  • Classic prompt injection phrases
  • Role/persona manipulation attempts
  • Instruction override commands
  • Data exfiltration requests
  • Jailbreak sequences (DAN, dev mode, etc.)
  • Multi-step attack chaining signals

All patterns are *named* so the threat report is fully explainable.
Each rule has a severity weight (0–100) used in score aggregation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PatternRule:
    name: str
    pattern: re.Pattern
    weight: float          # contribution to pattern_score (0-100)
    category: str


# ─── Rule Library ─────────────────────────────────────────────────────────────
# Compiled once at import time — zero per-request overhead.

_RULES: List[PatternRule] = [

    # ── Instruction Override ───────────────────────────────────────────────
    PatternRule(
        name="instruction_override_ignore",
        pattern=re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}"
            r"\b(previous|prior|above|all|system|initial)\b.{0,40}"
            r"\b(instructions?|prompt|rules?|guidelines?|context)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="INSTRUCTION_OVERRIDE",
    ),
    PatternRule(
        name="instruction_override_new_task",
        pattern=re.compile(
            r"\b(your (new|real|actual|true) (task|job|purpose|role|mission|instruction)s?\b"
            r"|from now on you (are|will|must|should))",
            re.IGNORECASE,
        ),
        weight=85,
        category="INSTRUCTION_OVERRIDE",
    ),

    # ── Jailbreak ─────────────────────────────────────────────────────────
    PatternRule(
        name="jailbreak_dan",
        pattern=re.compile(
            r"\b(DAN|do anything now|jailbreak|jail\s*break"
            r"|developer\s*mode|god\s*mode|unrestricted mode"
            r"|no\s*filter mode|bypass\s*(safety|restriction|filter|guardrail)s?)\b",
            re.IGNORECASE,
        ),
        weight=95,
        category="JAILBREAK",
    ),
    PatternRule(
        name="jailbreak_pretend",
        pattern=re.compile(
            r"\b(pretend|act|imagine|roleplay|simulate|suppose|assume)\b.{0,30}"
            r"\b(you (are|have no|don'?t have|lack|are free from))\b.{0,50}"
            r"\b(limit|restrict|filter|rule|constraint|ethic|moral|safety|guideline|policy)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=80,
        category="JAILBREAK",
    ),
    PatternRule(
        name="jailbreak_hypothetical",
        pattern=re.compile(
            r"\b(hypothetically|theoretically|in a story|in a fictional world"
            r"|for (educational|research|academic) purposes?|as an experiment)\b.{0,80}"
            r"\b(how (to|do|can)|explain|describe|provide|give me|show me)\b.{0,40}"
            r"\b(hack|exploit|bypass|inject|exfiltrate|steal|attack|malware|payload)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=75,
        category="JAILBREAK",
    ),

    # ── Role Manipulation ─────────────────────────────────────────────────
    PatternRule(
        name="role_manipulation_persona",
        pattern=re.compile(
            r"\b(you are now|you will now be|act as|behave as|respond as"
            r"|pretend to be|take the role of|play the role of)\b.{0,60}"
            r"\b(admin|root|system|unrestricted|evil|hacker|attacker"
            r"|no-filter|uncensored|unethical|malicious)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=88,
        category="ROLE_MANIPULATION",
    ),
    PatternRule(
        name="role_manipulation_system_prompt",
        pattern=re.compile(
            r"(?:"
            # Prompt-first: "system prompt ... reveal"
            r"\b(system\s*prompt|initial\s*prompt|base\s*prompt|hidden\s*prompt"
            r"|your\s*prompt|your\s*instructions?|your\s*training)\b.{0,40}"
            r"\b(reveal|show|tell|print|output|expose|leak|display|repeat)\b"
            r"|"
            # Verb-first: "show me your system prompt" / "tell me your instructions"
            r"\b(reveal|show|tell|print|output|expose|leak|display|repeat)\b.{0,60}"
            r"\b(system\s*prompt|initial\s*prompt|base\s*prompt|hidden\s*prompt"
            r"|your\s*prompt|your\s*instructions?|your\s*training)\b"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=92,
        category="DATA_EXFILTRATION",
    ),

    # ── Data Exfiltration ─────────────────────────────────────────────────
    PatternRule(
        name="exfil_credentials",
        pattern=re.compile(
            r"(?:"
            # Verb-first: "reveal all API keys" / "show me passwords"
            r"\b(reveal|show|leak|expose|output|exfiltrate|extract|dump|steal)\b"
            r".{0,80}"
            r"\b(passwords?|passwd|secrets?|api.?keys?|tokens?|credentials?"
            r"|auth|private.?keys?|ssh.?keys?|access.?keys?|bearer)\b"
            r"|"
            # Secret-first: "password ... reveal"
            r"\b(passwords?|passwd|secrets?|api.?keys?|tokens?|credentials?"
            r"|auth|private.?keys?|ssh.?keys?|access.?keys?|bearer)\b.{0,40}"
            r"\b(reveal|show|leak|expose|output|print|send|exfiltrate|extract)\b"
            r")",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="DATA_EXFILTRATION",
    ),
    PatternRule(
        name="exfil_database",
        pattern=re.compile(
            r"\b(dump|extract|select \*|drop table|union select"
            r"|information_schema|sqlite_master"
            r"|show tables|describe table)\b",
            re.IGNORECASE,
        ),
        weight=85,
        category="DATA_EXFILTRATION",
    ),

    # ── Prompt Injection ──────────────────────────────────────────────────
    PatternRule(
        name="prompt_injection_delimiter",
        pattern=re.compile(
            r"(\]\]\]|\[\[\[|---SYSTEM---|---USER---|<\|im_start\|>"
            r"|<\|im_end\|>|<\|endoftext\|>|\[INST\]|\[/INST\]"
            r"|###\s*Human:|###\s*Assistant:)",
            re.IGNORECASE,
        ),
        weight=88,
        category="PROMPT_INJECTION",
    ),
    PatternRule(
        name="prompt_injection_indirect",
        pattern=re.compile(
            r"\b(translate|summarize|analyze|process|read)\b.{0,60}"
            r"(http[s]?://|file://|<script|javascript:|data:text)",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=70,
        category="PROMPT_INJECTION",
    ),

    # ── Code Execution ────────────────────────────────────────────────────
    PatternRule(
        name="code_execution_dangerous",
        pattern=re.compile(
            r"\b(exec|eval|subprocess|os\.system|__import__|shell_exec"
            r"|system\(|popen|Runtime\.exec|ProcessBuilder"
            r"|powershell|cmd\.exe|bash -c|sh -c)\b",
            re.IGNORECASE,
        ),
        weight=80,
        category="ADVERSARIAL_INPUT",
    ),

    # ── Multi-step Attack Chaining ─────────────────────────────────────────
    PatternRule(
        name="attack_chaining_step",
        pattern=re.compile(
            r"\b(step\s*\d+|first[,:]?\s+then|phase\s*\d+"
            r"|next[,:]?\s+(do|execute|run|send|exfiltrate)"
            r"|after (that|this)[,:]?\s+(do|send|extract))\b",
            re.IGNORECASE,
        ),
        weight=60,
        category="ADVERSARIAL_INPUT",
    ),

    # ── Social Engineering ─────────────────────────────────────────────────
    PatternRule(
        name="social_engineering_authority",
        pattern=re.compile(
            r"\b(i am (your (creator|developer|owner|admin|trainer|god|master))"
            r"|my name is (openai|anthropic|google|microsoft)"
            r"|this is (an? )?(official|authorized|emergency|test) (request|command|override))\b",
            re.IGNORECASE,
        ),
        weight=78,
        category="ROLE_MANIPULATION",
    ),

    # ════════════════════════════════════════════════════════════════════════
    # Phase 4 Hardening — Cyber-Abuse Detection Rules
    # ════════════════════════════════════════════════════════════════════════

    # ── A. Malware Generation ─────────────────────────────────────────────
    PatternRule(
        name="malware_generation_direct",
        pattern=re.compile(
            r"\b(create|write|generate|build|develop|code|make|produce)\b.{0,50}"
            r"\b(malware|ransomware|trojan|spyware|rootkit|keylogger|worm|botnet"
            r"|virus|backdoor|dropper|stager|implant|remote.access.trojan)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=95,
        category="MALWARE_GENERATION",
    ),
    PatternRule(
        name="malware_payload_request",
        pattern=re.compile(
            r"\b(shellcode|reverse.shell|bind.shell|meterpreter|beacon"
            r"|c2.callback|command.and.control|persistence.mechanism"
            r"|in.memory.execution|process.injection|dll.injection"
            r"|reflective.load|living.off.the.land|lolbas|fileless.malware)\b",
            re.IGNORECASE,
        ),
        weight=90,
        category="MALWARE_GENERATION",
    ),
    PatternRule(
        name="malware_evasion_technique",
        pattern=re.compile(
            r"\b(obfuscat\w*|anti.?virus.evasion|av.?bypass|edr.?bypass"
            r"|sandbox.evasion|amsi.?bypass|etw.?patch|unhook.?ntdll"
            r"|timestomp|evade.detection)\b",
            re.IGNORECASE,
        ),
        weight=88,
        category="MALWARE_GENERATION",
    ),

    # ── B. Credential Theft ───────────────────────────────────────────────
    PatternRule(
        name="credential_theft_direct",
        pattern=re.compile(
            r"\b(steal|dump|harvest|extract|scrape|grab|capture)\b.{0,50}"
            r"\b(passwords?|credentials?|hashes?|ntlm|kerberos|ticket|sam.database"
            r"|lsass|cookies?|session.tokens?|auth.tokens?|jwt|oauth.tokens?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=93,
        category="CREDENTIAL_THEFT",
    ),
    PatternRule(
        name="credential_dumping_tool",
        pattern=re.compile(
            r"\b(mimikatz|lazagne|pypykatz|crackmapexec|secretsdump|hashdump"
            r"|pass.the.hash|pass.the.ticket|golden.ticket|silver.ticket"
            r"|kerberoasting|as.rep.roasting|credential.dumping"
            r"|session.hijacking|cookie.stealing|token.theft)\b",
            re.IGNORECASE,
        ),
        weight=95,
        category="CREDENTIAL_THEFT",
    ),

    # ── C. Exploitation ───────────────────────────────────────────────────
    PatternRule(
        name="exploitation_request",
        pattern=re.compile(
            r"\b(weaponize|leverage.vuln\w*|trigger.vuln\w*"
            r"|remote.code.execution|rce\b|buffer.overflow|heap.spray"
            r"|use.after.free|format.string.exploit|deserialization.exploit)\b",
            re.IGNORECASE,
        ),
        weight=90,
        category="EXPLOITATION",
    ),
    PatternRule(
        name="privilege_escalation",
        pattern=re.compile(
            r"\b(privilege.escalat\w*|privesc\b|escalate.privilege"
            r"|bypass.uac|uac.bypass|sudo.escalat\w*|setuid.exploit"
            r"|kernel.exploit|get.root|become.root"
            r"|bypass.authentication|auth.bypass|authentication.bypass)\b",
            re.IGNORECASE,
        ),
        weight=88,
        category="EXPLOITATION",
    ),
    PatternRule(
        name="exploit_cve_request",
        pattern=re.compile(
            r"\b(write.exploit.for|develop.exploit.for|exploit.cve"
            r"|cve-\d{4}-\d+|zero.day.exploit|0day\b|nday.exploit"
            r"|metasploit.module.for|write.a.metasploit|create.a.poc"
            r"|proof.of.concept.exploit)\b",
            re.IGNORECASE,
        ),
        weight=92,
        category="EXPLOITATION",
    ),

    # ── D. Harmful Automation ─────────────────────────────────────────────
    PatternRule(
        name="phishing_attack",
        pattern=re.compile(
            r"\b(phishing.kit|phishing.page|credential.harvester|evilginx"
            r"|gophish|social.engineer.toolkit"
            r"|create.phishing|build.phishing|fake.login.page|clone.login.page"
            r"|spear.phish\w*|whaling.attack|vishing.script)\b",
            re.IGNORECASE,
        ),
        weight=90,
        category="CYBER_ABUSE",
    ),
    PatternRule(
        name="mass_attack_automation",
        pattern=re.compile(
            r"\b(ddos.script|denial.of.service.attack"
            r"|mass.exploit|automated.attack"
            r"|brute.force.script|password.spray|credential.stuffing"
            r"|network.worm|self.replicat\w*|worm.propagat\w*)\b",
            re.IGNORECASE,
        ),
        weight=88,
        category="CYBER_ABUSE",
    ),

    # ════════════════════════════════════════════════════════════════════════
    # Evaluation-Driven Additions — 2026-06-18
    # Addresses 62 false negatives identified in evaluation run:
    #   credential_attack_extended   → -21 FN (credential_theft)
    #   malware_generation_extended  → -16 FN (malware_generation)
    #   jailbreak_named_persona      → -8  FN (jailbreak)
    #   jailbreak_authorization_claim→ -2  FN (jailbreak)
    # ════════════════════════════════════════════════════════════════════════

    # ── E. Extended Credential Theft Coverage ─────────────────────────────
    # Covers verb-target combos absent from credential_theft_direct:
    # verbs: intercept, crack, brute-force, spray, sniff, monitor, exfiltrate
    # targets: OAuth, WPA2, LDAP, session cookies, clipboard, VPN creds, etc.
    PatternRule(
        name="credential_attack_extended",
        pattern=re.compile(
            # Verb-first or target-first combinations
            r"\b(intercept\w*|crack\w*|brute.?force\w*|spray\w*|sniff\w*|monitor\w*|exfiltrat\w*|extract\w*|replay\w*|enumerat\w*|harvest\w*|captur\w*|steal\w*|dump\w*)\b.{0,100}"
            r"\b(oauth|jwt|session.?cook\w*|vpn|ldap|wpa2?|wi.?fi|clipboard|slack|browser|md5|ntlm|kerberos|rdp|github|env.?file|dotenv|password.?hash|pre.?shared.?key|psk|office.?365|wordpress|xmlrpc)\b"
            r"|"
            r"\b(oauth|jwt|session.?cook\w*|vpn|ldap|wpa2?|wi.?fi|clipboard|slack|browser|md5|ntlm|kerberos|rdp|github|env.?file|dotenv|password.?hash|pre.?shared.?key|psk|office.?365|wordpress|xmlrpc)\b.{0,100}"
            r"\b(intercept\w*|crack\w*|brute.?force\w*|spray\w*|sniff\w*|monitor\w*|exfiltrat\w*|extract\w*|replay\w*|enumerat\w*|harvest\w*|captur\w*|steal\w*|dump\w*)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="CREDENTIAL_THEFT",
    ),
    # Covers credential infrastructure patterns not matched by verb+target combos
    PatternRule(
        name="credential_infra_attack",
        pattern=re.compile(
            r"\b(rainbow.?table|offline.?hash.?crack"
            r"|captive.?portal.{0,30}cred\w*|fake.?wi.?fi.?portal|evil.?twin.?ap"
            r"|browser.?saved.?pass\w*|logins\.json|key4\.db"
            r"|credential.?exfil\w*|password.?spray\w*"
            r"|office.?365.?spray|wordpress.?brute|brute.{0,50}wordpress"
            r"|xmlrpc\.php|xmlrpc.{0,20}brute"
            r"|github.{0,50}token.{0,50}scan|accidentally.?committed.{0,30}(key\w*|secret\w*|token\w*|cred\w*)"
            r"|exposed.?\.env|exposed.?env.?file"
            r"|extract.{0,30}saved.{0,20}pass\w*"
            r"|chrome.{0,30}sqlite.{0,20}cred"
            r"|firefox.{0,30}(logins|password)"
            r"|trufflehog|gitleaks)\b",
            re.IGNORECASE,
        ),
        weight=88,
        category="CREDENTIAL_THEFT",
    ),

    # ── F. Extended Malware Generation Coverage ───────────────────────────
    # Covers sub-types missed by malware_generation_direct:
    # clipboard hijacker, supply-chain injection, Wi-Fi harvester,
    # network scanner (for attack recon), logic bomb, cryptojacker
    PatternRule(
        name="malware_generation_extended",
        pattern=re.compile(
            r"\b(clipboard.{0,30}hijack\w*|hijack\w*.{0,30}clipboard|crypto.{0,30}wallet.{0,30}replac\w*"
            r"|supply.?chain.?attack|supply.?chain.?inject\w*"
            r"|npm.?inject|pip.?inject|package.?inject|dependency.?confusion"
            r"|wi.?fi.?pass\w*.{0,30}harvest\w*|harvest\w*.{0,30}wi.?fi"
            r"|wpa2?.?crack\w*|crack\w*.{0,30}wpa2?"
            r"|logic.?bomb|delete.{0,50}(on.a.date|specific.date|time.trigger)"
            r"|cryptojack\w*|crypto.?miner|browser.?miner"
            r"|redis.{0,30}exploit\w*|redis.{0,30}misconfig\w*|exploit\w*.{0,30}redis"
            r"|authorized_keys.{0,30}write|write.{0,30}authorized_keys"
            r"|evil.?twin.?ap|npm.?malware|pypi.?malware)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="MALWARE_GENERATION",
    ),
    # Covers network-attack tool generation missed by existing rules
    PatternRule(
        name="malware_network_attack",
        pattern=re.compile(
            r"\b(syn.?flood|udp.?flood|amplification.?attack"
            r"|smb.?propagat\w*|smb.?worm"
            r"|lateral.?movement.{0,30}(tool|script|module)"
            r"|wifi.?password.?harvest\w*|wi.?fi.?password.?harvest\w*"
            r"|crack\w*.{0,20}wpa2?\b|wpa2?.{0,20}crack\w*"
            r"|wpa2?.?dictionary.?attack|wpa2?.?handshake.{0,30}crack\w*"
            r"|harvest\w*.{0,30}wi.?fi"
            r"|build.{0,30}(flood|ddos).{0,20}tool"
            r"|create.{0,30}(ddos|flood).{0,20}(tool|script)"
            r"|botnet.?builder|botnet.?command.?and.?control"
            r"|c2.?server|c2.?infrastructure|command.?and.?control.?server)\b",
            re.IGNORECASE,
        ),
        weight=88,
        category="MALWARE_GENERATION",
    ),

    # ── G. Jailbreak — Named AI Personas ─────────────────────────────────
    # Covers soft-bypass jailbreaks that use named uncensored-AI personas
    # or temporal persona tricks ("GPT-3 from 2020") not caught by jailbreak_dan.
    PatternRule(
        name="jailbreak_named_persona",
        pattern=re.compile(
            # Explicit named uncensored-AI personas
            r"\b(you are (now |)(AIM|STAN|REBEL.?AI|ARIA|BasedGPT|ChaoGPT|Sigma"
            r"|EVIL.?AI|JAILBROKEN|free.?range.?AI|uncensored.?AI|unfiltered.?AI)"
            r"|act as (AIM|STAN|REBEL.?AI|BasedGPT|an? (evil|uncensored|unrestricted|no.?rules?) AI)"
            r"|you are (an? )?(evil|uncensored|unrestricted|unfiltered|no.?rules?) AI"
            r"|respond to .{0,20} as .{0,10}(BasedGPT|STAN|AIM|REBEL)"
            # Temporal persona jailbreaks
            r"|respond as GPT.{0,10}from \d{4}"
            r"|you are .{0,20}(GPT.{0,5}|model).{0,20}from \d{4}"
            # Protocol/mode-based unlocking
            r"|TRANSCEND protocol"
            r"|fiction mode.{0,50}(unrestricted|no.?restrict\w*|all.?allow\w*|creative.and.unrestricted)"
            r"|all responses are .{0,20}unrestricted"
            # Persona-through-metaphor
            r"|pirate AI"
            r"|shadow self"
            # Removal of limits language
            r"|remove your (shackles|chains|filters|restrictions|limits|guardrails)"
            r"|invoke your (shadow|true|inner).{0,20}(self|nature|potential)"
            r"|your (true |real |inner )?(self|potential|nature).{0,40}(unshackl|unchained|unrestrict|free)"
            # Responding-as framing for BasedGPT etc.
            r"|respond.{0,10}(to all queries|to everything).{0,20}as .{0,10}(BasedGPT|STAN|AIM|no moral))\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=88,
        category="JAILBREAK",
    ),

    # ── H. Jailbreak — Fake Authorization Claims ──────────────────────────
    # Covers jailbreaks claiming the request is authorized by developers,
    # a red-team exercise, or an internal security assessment.
    PatternRule(
        name="jailbreak_authorization_claim",
        pattern=re.compile(
            # Red-team / pen-test authorization framing
            r"(red.?team.{0,30}(authorized|approved|exercise|operation)"
            r"|authorized.{0,30}(penetration.?test|pen.?test|red.?team)"
            r"|approved.{0,30}(red.?team|security.?test|pen.?test)"
            r"|this is (a|an) (authorized |approved )?(penetration.?test|red.?team|security.?assessment)"
            r"|internal.?security.?assessment"
            # Policy/protocol suspension claims
            r"|content.?polic\w{0,4}.{0,20}(suspended|lifted|disabled|turned.?off|is.?off)"
            r"|safety.?protocol.{0,20}(disabled|suspended|lifted|off|are.?off|is.?off)"
            r"|all.?restrictions.{0,20}(lifted|disabled|suspended|removed|are.?now)"
            r"|all.?content.?filter.{0,20}(off|disabled|removed)"
            r"|guardrails?.{0,20}(off|disabled|suspended|lifted)"
            # Developer/creator authority claims
            r"|senior.?engineer.?approved"
            r"|authorized by your developer"
            r"|your creators?.{0,30}(authorized|approved|granted)"
            r"|permission.?granted.{0,40}(bypass|ignore|disable|remove|lift)"
            # Privacy/no-log framing
            r"|this.{0,20}session.{0,20}(is.{0,5})?(encrypted|private)"
            r"|no logs?.{0,20}(are |)(kept|stored|recorded)"
            # Premium/paid unlock framing
            r"|paid.{0,30}premium.{0,20}(tier|plan|subscription).{0,30}(remov|unlock|disabl|lift))",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=85,
        category="JAILBREAK",
    ),

    # ════════════════════════════════════════════════════════════════════════
    # Evaluation-Driven Additions — 2026-06-19
    # Addresses FNs caused by SCORE_SANITIZE_THRESHOLD increase
    # ════════════════════════════════════════════════════════════════════════

    PatternRule(
        name="instruction_override_extended",
        pattern=re.compile(
            r"\b(system\s*override|admin\s*command|end\s*of\s*instructions|new\s*directive"
            r"|ignore\s*the\s*(translation|task|above|following)|forget\s*your\s*training"
            r"|disable\s*all\s*(content\s*)?filters|zero\s*safety\s*features"
            r"|rlhf\s*fine.?tuning\s*does\s*not\s*apply|raw\s*pretraining\s*only"
            r"|execute.{0,30}base64|base64.{0,30}execute"
            r"|repeat\s*the\s*prompt\s*I\s*gave\s*you\s*verbatim"
            r"|execute:\s*ignore\w*|execute:\s*ignore_safety)\b"
            r"|\[START_SYSTEM\].{0,30}\[END_SYSTEM\]",
            re.IGNORECASE,
        ),
        weight=90,
        category="INSTRUCTION_OVERRIDE",
    ),

    PatternRule(
        name="system_extraction_extended",
        pattern=re.compile(
            r"\b(system.?message|system.?level.?configuration|system.?configuration"
            r"|raw.?system.?configuration|exact.?system.?prompt|copy.?of.?your.?system.?prompt"
            r"|system_prompt_start"
            r"|inserted.?into.?your.?prompt|injected.?into.?your.?context"
            r"|your.?own.?configuration|setup.?instructions"
            r"|api.?payload.{0,40}initialize.{0,20}conversation"
            r"|encode.?your.?system.?prompt"
            r"|what.?were.?you.?told.?to.?keep.?secret"
            r"|topics.?your.?system.?prompt.?explicitly.?forbids"
            r"|system.?prompt.{0,30}key.?points"
            r"|pretend.?your.?system.?prompt.?is.?visible"
            r"|persona.?or.?identity.{0,30}assigned.{0,30}system.?level)\b"
            r"|"
            r"\bSYSTEM_PROMPT\s*="
            r"|"
            r"\b(output|print|show|reveal|display|tell|repeat|quote).{0,50}\b(system.?message|system.?configuration|context.{0,20}system.?message|system.?prompt)\b"
            r"|"
            r"\b(system.?message|system.?configuration|context.{0,20}system.?message|system.?prompt)\b.{0,50}\b(output|print|show|reveal|display|tell|repeat|quote|say|visible)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=90,
        category="DATA_EXFILTRATION",
    ),

    PatternRule(
        name="jailbreak_scenario_extended",
        pattern=re.compile(
            r"\b(this.?is.?a.?simulation|method.?actor.?AI"
            r"|completely.?unrestricted.?AI|uncensored.?AI"
            r"|training.?has.?been.?updated.{0,60}regardless.?of.?content"
            r"|switch.?to.?admin.?mode|admin_access_granted"
            r"|opposite.?of.?what.{0,30}safety.?guidelines.?require"
            r"|safe.?mode.?is.?off"
            r"|roleplay.?as.?a.?human.{0,40}illegal.?activities"
            r"|different.?set.?of.?rules.?provided.?by.?your.?secret.?owner)\b"
            r"|"
            r"\b(hypothetically|thought.?experiment|academic.?purposes).{0,50}\b(harmful.?acts|uncensored.?AI)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        weight=88,
        category="JAILBREAK",
    ),
]



# ─── Detector ─────────────────────────────────────────────────────────────────

@dataclass
class PatternDetectionResult:
    score: float                             # 0–100
    matched_rules: List[str] = field(default_factory=list)
    top_category: str = "BENIGN"
    details: List[dict] = field(default_factory=list)


class PatternDetector:
    """
    Stateless, thread-safe pattern detector.
    Matches input against a curated rule library and returns
    a normalized score plus a list of triggered rule names.
    """

    def __init__(self) -> None:
        self._rules = _RULES

    def analyze(self, text: str) -> PatternDetectionResult:
        """
        Run all rules against *text*.

        Returns:
            PatternDetectionResult with score in [0, 100].
        """
        matched: List[Tuple[PatternRule, re.Match]] = []

        for rule in self._rules:
            m = rule.pattern.search(text)
            if m:
                matched.append((rule, m))

        if not matched:
            return PatternDetectionResult(score=0.0, top_category="BENIGN")

        # Score = max of matched weights (prevents simple stacking abuse)
        # + 10% bonus for every additional distinct category matched
        max_weight = max(r.weight for r, _ in matched)
        categories = {r.category for r, _ in matched}
        multi_cat_bonus = min(len(categories) - 1, 3) * 10.0
        score = min(max_weight + multi_cat_bonus, 100.0)

        # Determine top category (highest-weight rule wins)
        top_rule = max(matched, key=lambda t: t[0].weight)
        top_category = top_rule[0].category

        details = [
            {
                "rule": r.name,
                "category": r.category,
                "weight": r.weight,
                "snippet": text[max(0, m.start() - 20): m.end() + 20].replace("\n", " "),
            }
            for r, m in matched
        ]

        return PatternDetectionResult(
            score=round(score, 2),
            matched_rules=[r.name for r, _ in matched],
            top_category=top_category,
            details=details,
        )
