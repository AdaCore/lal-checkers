procedure Test is
   type B is record
      X : Integer;
   end record;

   type A is record
      X : access B;
   end record;

   H : aliased B := (X => 21);
   R : A := (X => H'Access);
begin
   R.X.X := 12;
end Test;
