---
title: Access Management
kind: site
site_path: /docs/access-management
---

**A guide to requesting, receiving and using access** to BAPa's licensed analysis application.

## Why access is controlled

Most of the site is open to anyone — the research chat, the publication library, the interview and
the background pages. **The analysis application is not.** It is licensed, and access is granted
individually, both to limit it to approved people and to establish which named account is using it.

The gate does **not** change where laboratory data goes (analysis runs entirely in the browser
whether signed in or not, and the file is never uploaded — see the security and privacy
documentation), and it does not record what a user does once signed in.

## Requesting access

1. **Sign in once.** Go to the site and press **Get Started!**, or open the **App** tab. Complete
   sign-in with a Google account. This creates the account; there is no separate registration form.
2. **Land on a pending screen** reading "Your access is pending approval," showing the email
   address signed in with.
3. **Write to the project administrator.** This step is manual and matters: creating the account
   does not notify anyone. BAPa sends no email at all and has no mail service — access to a
   licensed application is a considered, individual decision made by a person, not an automated
   queue.

**Write to Dr. George Cembrowski, cembr001@gmail.com**, including: the exact email address signed
in with; who the requester is (name, role, institution); what the application will be used for
(the clinical or research purpose, and the kind of laboratory data expected); and anything relevant
to the licence (individual use, a department, or an institutional evaluation). If evaluating BAPa on
behalf of an institution, say so, and send the security and privacy documentation to IT and privacy
teams first.

## Onboarding

There are no credentials to issue and nothing arrives by email. An administrator sets a role in the
Clerk dashboard by hand — **User** grants the analysis application; **Admin** additionally grants
management of the publication library. Most people need User. Once approved, reload the page — no
new sign-in is needed, and the change takes effect within about a minute. Access can be withdrawn
the same way.

## Using the tool once approved

Go to the site, press **Get Started!** or the **App** tab, sign in with Google if the session has
lapsed, and the analysis application loads (around a minute on first use per machine/browser while
the analysis engine downloads). Step 1 is Load Dataset — see the analysis tool documentation for
the full six-step walkthrough.

## Troubleshooting

- **"This app requires access from an administrator"** — not signed in yet; press Continue to sign
  in.
- **"Your access is pending approval"** — signed in but no role attached. Either nobody has been
  asked yet (email the administrator), it was just granted and the page needs a reload, or something
  is misconfigured (report it if a confirmed grant still shows this after a few minutes).
- **"Failed to load analysis engine"** — usually a network problem; on a hospital or corporate
  network, check that `cdn.jsdelivr.net` isn't blocked.
- **"The analysis tool hit an error"** — the application failed, not the user's access; the file
  never left the browser, and an anonymous crash report with no data content was sent. Press "Start
  the tool over."
- **No sign-out button exists once approved** — on a shared workstation, close the browser or clear
  cookies.

Any other issue: write to cembr001@gmail.com with what was being done, what was expected, and what
happened instead.
