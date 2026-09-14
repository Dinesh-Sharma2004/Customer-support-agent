# Intent Taxonomy

The frozen taxonomy has six classes:

| Intent | Definition | Belongs Here | Does Not Belong | Routing Implication | Ambiguity Boundary |
|---|---|---|---|---|---|
| Order & Delivery Issues | Order status, tracking, late, missing, delivered-not-received, shipment issues. | "Where is my package?", "It says delivered but nothing arrived." | Refund-only complaints, return instructions. | Can be auto-handled only with strong evidence and confidence. | Replacement shipments may overlap with returns. |
| Other / Human Review | Ambiguous, unsupported, feedback, multi-issue, weak-signal, or non-operational messages. | Praise, vague complaints, combined issues, unclear asks. | Clear delivery, return, refund, Prime, or account-security cases. | Usually escalated or reviewed. | Must not become a lazy garbage class; it is a safety class for unclear routing. |
| Prime & Membership | Prime subscription, membership benefits, Prime video/music/shipping feature issues. | "Prime isn't working", "Cancel membership." | Delivery issue from a Prime order unless membership itself is the issue. | Auto-handle only when evidence is strong and no account/private action is needed. | Prime keywords often co-occur with delivery complaints. |
| Refunds & Financials | Refund status, charges, billing, credits, reimbursement, payment complaints. | "Where is my refund?", "I was charged twice." | Return initiation without a money question. | Higher caution because money/account facts cannot be invented. | Returns often imply refunds but are not identical. |
| Account Security & Private Support | Login, password, verification, hacked account, unauthorized access, private account help. | "My account was hacked", "I can't sign in." | Generic Prime/account preference questions without security/privacy risk. | Always escalate in the policy. | Login trouble can be convenience or security; policy treats it cautiously. |
| Returns & Exchanges | Returning, replacing, exchanging, sending items back. | "I want to return this", "Can I exchange it?" | Refund status after return completion. | Auto-handle only with strong historical support and safe wording. | Damaged delivery plus return request may be multi-issue. |

Six classes were chosen to balance operational routing with enough data per class. Clustering/EDA is exploratory: it helps discover common issue families, but it does not automatically define final labels. The final taxonomy is a human engineering decision frozen before evaluation.

Next: [Classifier](04_classifier.md).
