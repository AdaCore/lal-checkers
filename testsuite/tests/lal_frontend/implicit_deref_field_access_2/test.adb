procedure Test is
   type Type_D is record
      e: Integer;
   end record;

   type Type_C is record
      d: access Type_D;
   end record;

   type Type_B is record
      c: access Type_C;
   end record;

   type Type_A is record
      b: access Type_B;
   end record;

   a : Type_A;
begin
   a.b.c.d.e := a.b.c.d.e;
end Ex1;
