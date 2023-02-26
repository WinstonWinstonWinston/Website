# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 130

title: Contact
subtitle: Get in touch

content:
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      text: |-
        If you have any questions or want to get into contact, feel free to send me an email!
      # Contact (add or remove contact options as necessary)
      email: h.sully2015@gmail.com
      phone: 801-641-9157
      address:
        street: 918 E 500 S Unit C
        city: Salt Lake City
        region: UT
        postcode: '84102'
        country: United States
        country_code: US
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      form:
        provider: netlify
        formspree:
          id:
        netlify:
          # Enable CAPTCHA challenge to reduce spam?
          captcha: true
    design:
      columns: '2'
